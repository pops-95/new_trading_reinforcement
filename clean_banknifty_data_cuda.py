#!/usr/bin/env python3
"""
CUDA-assisted streaming cleaner for BANKNIFTY processed option features.

Default input:
    data/banknifty_processed_stream/banknifty_option_features_model_ready.parquet

Default output:
    data/banknifty_cleaned/

Main tasks:
- stream parquet batches to avoid OOM
- drop deprecated fully-empty columns such as theta and vega
- preserve genuine volume=0 and open_interest=0
- invalid OHLC <=0 -> NaN
- invalid negative volume/OI -> NaN
- +/-inf -> NaN
- add individual_oi_available
- recompute TRUE consecutive-1-minute:
    premium_change_1m
    option_return_1m
    volume_change_1m
    oi_change_1m
    oi_pct_change_1m
    price_oi_change
    price_return_x_oi_pct_change
    iv_change_1m
    delta_change_1m
- no forward-fill/back-fill
- CUDA/CuPy used for vectorized numeric cleanup/diff arithmetic
- DuckDB combines parts without loading full dataset into RAM
- writes cleaning_report.json and feature_manifest.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import cupy as cp
except Exception as exc:
    raise SystemExit(
        "CuPy is required. Install cupy-cuda12x or cupy-cuda11x.\n"
        f"Import error: {exc}"
    )

try:
    import duckdb
except Exception as exc:
    raise SystemExit(
        "DuckDB is required. Install with: pip install duckdb\n"
        f"Import error: {exc}"
    )

DEPRECATED_COLUMNS = {"theta", "vega"}
PRICE_COLUMNS = ("open", "high", "low", "close")
GENERIC_NUMERIC_CHECK = (
    "spot_close", "strike", "implied_volatility", "delta",
    "gamma", "theta_per_day", "vega_per_pct",
)

def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)

def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)

def release_memory() -> None:
    gc.collect()
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

def cuda_info() -> Dict[str, Any]:
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    name = props.get("name", b"unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="ignore")
    return {
        "device_id": int(dev.id),
        "name": str(name),
        "compute_capability": f"{int(props.get('major',0))}.{int(props.get('minor',0))}",
        "free_gb": round(free_b / (1024**3), 3),
        "total_gb": round(total_b / (1024**3), 3),
    }

def duckdb_connection(threads: int, memory_limit_gb: float, temp_dir: Path):
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(threads)}")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA memory_limit='{float(memory_limit_gb):.1f}GB'")
    escaped = str(temp_dir).replace("'", "''")
    con.execute(f"PRAGMA temp_directory='{escaped}'")
    return con

def gpu_clean_numeric_array(arr, positive_only=False, nonnegative_only=False):
    x = cp.asarray(arr, dtype=cp.float64)
    x = cp.where(cp.isfinite(x), x, cp.nan)
    if positive_only:
        x = cp.where(x > 0, x, cp.nan)
    elif nonnegative_only:
        x = cp.where(x >= 0, x, cp.nan)
    out = cp.asnumpy(x)
    del x
    return out

def gpu_safe_diff(current, previous, valid_pair):
    cur = cp.asarray(current, dtype=cp.float64)
    prev = cp.asarray(previous, dtype=cp.float64)
    valid = cp.asarray(valid_pair, dtype=cp.bool_)
    out = cp.where(
        valid & cp.isfinite(cur) & cp.isfinite(prev),
        cur - prev,
        cp.nan,
    )
    result = cp.asnumpy(out)
    del cur, prev, valid, out
    return result

def gpu_safe_pct_change(current, previous, valid_pair):
    cur = cp.asarray(current, dtype=cp.float64)
    prev = cp.asarray(previous, dtype=cp.float64)
    valid = cp.asarray(valid_pair, dtype=cp.bool_)
    out = cp.where(
        valid & cp.isfinite(cur) & cp.isfinite(prev) & (prev != 0),
        (cur - prev) / prev,
        cp.nan,
    )
    result = cp.asnumpy(out)
    del cur, prev, valid, out
    return result

def gpu_mul(a, b):
    aa = cp.asarray(a, dtype=cp.float64)
    bb = cp.asarray(b, dtype=cp.float64)
    out = cp.where(cp.isfinite(aa) & cp.isfinite(bb), aa * bb, cp.nan)
    result = cp.asnumpy(out)
    del aa, bb, out
    return result

def clean_batch_numeric(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    drop_cols = [c for c in DEPRECATED_COLUMNS if c in x.columns]
    if drop_cols:
        x = x.drop(columns=drop_cols)

    if "timestamp" in x.columns:
        x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
    if "expiry_date" in x.columns:
        x["expiry_date"] = pd.to_datetime(x["expiry_date"], errors="coerce")

    for col in PRICE_COLUMNS:
        if col in x.columns:
            vals = pd.to_numeric(x[col], errors="coerce").to_numpy(float)
            x[col] = gpu_clean_numeric_array(vals, positive_only=True)

    for col in ("spot_close", "strike"):
        if col in x.columns:
            vals = pd.to_numeric(x[col], errors="coerce").to_numpy(float)
            x[col] = gpu_clean_numeric_array(vals, positive_only=True)

    for col in ("volume", "open_interest"):
        if col in x.columns:
            vals = pd.to_numeric(x[col], errors="coerce").to_numpy(float)
            x[col] = gpu_clean_numeric_array(vals, nonnegative_only=True)

    for col in GENERIC_NUMERIC_CHECK:
        if col in x.columns and col not in ("spot_close", "strike"):
            vals = pd.to_numeric(x[col], errors="coerce").to_numpy(float)
            x[col] = gpu_clean_numeric_array(vals)

    x["individual_oi_available"] = (
        x["open_interest"].notna().astype("int8")
        if "open_interest" in x.columns
        else np.int8(0)
    )
    return x.replace([np.inf, -np.inf], np.nan)

def recompute_true_1m_changes(
    df: pd.DataFrame,
    carry_state: Dict[str, Dict[str, Any]],
):
    x = df.copy()
    if x.empty:
        return x, carry_state

    x = x.sort_values(["groww_symbol", "timestamp"]).reset_index(drop=True)
    n = len(x)

    prev_ts = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    prev = {
        "close": np.full(n, np.nan),
        "volume": np.full(n, np.nan),
        "open_interest": np.full(n, np.nan),
        "implied_volatility": np.full(n, np.nan),
        "delta": np.full(n, np.nan),
    }

    grouped = x.groupby("groww_symbol", sort=False)
    prev_ts[:] = grouped["timestamp"].shift(1).to_numpy(dtype="datetime64[ns]")

    for col in prev:
        if col in x.columns:
            prev[col][:] = grouped[col].shift(1).to_numpy(float)

    first_idx = grouped.head(1).index
    for idx in first_idx:
        symbol = x.at[idx, "groww_symbol"]
        state = carry_state.get(symbol)
        if not state:
            continue
        prev_ts[idx] = np.datetime64(state["timestamp"])
        for col in prev:
            prev[col][idx] = state.get(col, np.nan)

    current_ts = x["timestamp"].to_numpy(dtype="datetime64[ns]")
    current_ns = current_ts.astype("int64")
    prev_ns = prev_ts.astype("int64")

    nat_int = np.datetime64("NaT").astype("datetime64[ns]").astype("int64")
    valid_ts = (current_ns != nat_int) & (prev_ns != nat_int)
    one_minute_ns = int(pd.Timedelta(minutes=1).value)
    valid_pair = valid_ts & ((current_ns - prev_ns) == one_minute_ns)

    if "close" in x.columns:
        cur = x["close"].to_numpy(float)
        x["premium_change_1m"] = gpu_safe_diff(cur, prev["close"], valid_pair)
        x["option_return_1m"] = gpu_safe_pct_change(cur, prev["close"], valid_pair)

    if "volume" in x.columns:
        x["volume_change_1m"] = gpu_safe_diff(
            x["volume"].to_numpy(float), prev["volume"], valid_pair
        )

    if "open_interest" in x.columns:
        cur_oi = x["open_interest"].to_numpy(float)
        x["oi_change_1m"] = gpu_safe_diff(
            cur_oi, prev["open_interest"], valid_pair
        )
        x["oi_pct_change_1m"] = gpu_safe_pct_change(
            cur_oi, prev["open_interest"], valid_pair
        )

    if "implied_volatility" in x.columns:
        x["iv_change_1m"] = gpu_safe_diff(
            x["implied_volatility"].to_numpy(float),
            prev["implied_volatility"],
            valid_pair,
        )

    if "delta" in x.columns:
        x["delta_change_1m"] = gpu_safe_diff(
            x["delta"].to_numpy(float),
            prev["delta"],
            valid_pair,
        )

    if "premium_change_1m" in x.columns and "oi_change_1m" in x.columns:
        x["price_oi_change"] = gpu_mul(
            x["premium_change_1m"].to_numpy(float),
            x["oi_change_1m"].to_numpy(float),
        )

    if "option_return_1m" in x.columns and "oi_pct_change_1m" in x.columns:
        x["price_return_x_oi_pct_change"] = gpu_mul(
            x["option_return_1m"].to_numpy(float),
            x["oi_pct_change_1m"].to_numpy(float),
        )

    x["true_1m_prev_available"] = valid_pair.astype("int8")

    last_rows = grouped.tail(1)
    for row in last_rows.itertuples(index=False):
        symbol = getattr(row, "groww_symbol")
        carry_state[symbol] = {
            "timestamp": getattr(row, "timestamp"),
            "close": getattr(row, "close", np.nan),
            "volume": getattr(row, "volume", np.nan),
            "open_interest": getattr(row, "open_interest", np.nan),
            "implied_volatility": getattr(row, "implied_volatility", np.nan),
            "delta": getattr(row, "delta", np.nan),
        }

    return x.replace([np.inf, -np.inf], np.nan), carry_state

def update_stats(stats: Dict[str, Any], df: pd.DataFrame):
    stats["rows"] += len(df)

    for src, key in (
        ("individual_oi_available", "individual_oi_available"),
        ("greeks_available", "greeks_available"),
        ("chain_oi_available", "chain_oi_available"),
        ("true_1m_prev_available", "true_1m_available"),
    ):
        if src in df.columns:
            stats[key] += int(df[src].fillna(0).sum())

    numeric = df.select_dtypes(include=[np.number])
    stats["inf_count"] += int(np.isinf(numeric.to_numpy()).sum())

    for col, count in df.isna().sum().items():
        stats["missing_counts"][col] = (
            stats["missing_counts"].get(col, 0) + int(count)
        )

def combine_parts(parts_glob, output_path, threads, memory_gb, temp_dir):
    con = duckdb_connection(threads, memory_gb, temp_dir)
    src = str(parts_glob).replace("'", "''")
    dst = str(output_path).replace("'", "''")
    con.execute(f"""
        COPY (
            SELECT *
            FROM read_parquet('{src}')
        )
        TO '{dst}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    con.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-file",
        default="data/banknifty_processed_stream/banknifty_option_features_model_ready.parquet",
    )
    ap.add_argument("--output-dir", default="data/banknifty_cleaned")
    ap.add_argument("--batch-rows", type=int, default=150000)
    ap.add_argument("--duckdb-threads", type=int, default=4)
    ap.add_argument("--duckdb-memory-gb", type=float, default=4.0)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)

    if not input_file.exists():
        raise SystemExit(f"Input parquet not found: {input_file}")

    if args.fresh and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "duckdb_temp"

    dev = cuda_info()
    print(
        f"CUDA device: {dev['name']} | CC={dev['compute_capability']} | "
        f"free={dev['free_gb']:.2f} GB | total={dev['total_gb']:.2f} GB"
    )
    atomic_json(dev, output_dir / "cuda_info.json")

    pf = pq.ParquetFile(input_file)
    print(f"Input rows: {pf.metadata.num_rows:,}")

    carry_state: Dict[str, Dict[str, Any]] = {}
    stats = {
        "rows": 0,
        "individual_oi_available": 0,
        "greeks_available": 0,
        "chain_oi_available": 0,
        "true_1m_available": 0,
        "inf_count": 0,
        "missing_counts": {},
    }

    for batch_idx, rb in enumerate(pf.iter_batches(batch_size=args.batch_rows)):
        part_path = parts_dir / f"part_{batch_idx:05d}.parquet"

        df = rb.to_pandas()
        print(f"[{batch_idx:05d}] rows={len(df):,}")

        df = clean_batch_numeric(df)
        df, carry_state = recompute_true_1m_changes(df, carry_state)
        df = df.replace([np.inf, -np.inf], np.nan)

        atomic_parquet(df, part_path)
        update_stats(stats, df)

        del df, rb
        release_memory()

    final_path = output_dir / "banknifty_option_features_clean.parquet"
    print("Combining cleaned parts...")
    combine_parts(
        parts_dir / "*.parquet",
        final_path,
        args.duckdb_threads,
        args.duckdb_memory_gb,
        temp_dir,
    )

    rows = stats["rows"]
    report = {
        "cuda": dev,
        "input_file": str(input_file),
        "output_file": str(final_path),
        "rows": int(rows),
        "individual_oi_available_fraction": (
            stats["individual_oi_available"] / rows if rows else None
        ),
        "greeks_available_fraction": (
            stats["greeks_available"] / rows if rows else None
        ),
        "chain_oi_available_fraction": (
            stats["chain_oi_available"] / rows if rows else None
        ),
        "true_1m_continuity_fraction": (
            stats["true_1m_available"] / rows if rows else None
        ),
        "numeric_inf_count": int(stats["inf_count"]),
        "dropped_columns": sorted(DEPRECATED_COLUMNS),
        "top_missing_columns": dict(
            sorted(
                stats["missing_counts"].items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:75]
        ),
        "settings": {
            "batch_rows": args.batch_rows,
            "duckdb_threads": args.duckdb_threads,
            "duckdb_memory_gb": args.duckdb_memory_gb,
        },
    }
    atomic_json(report, output_dir / "cleaning_report.json")

    schema = pq.ParquetFile(final_path).schema_arrow
    manifest = {}
    for field in schema:
        missing_count = int(stats["missing_counts"].get(field.name, 0))
        manifest[field.name] = {
            "dtype": str(field.type),
            "missing_count": missing_count,
            "missing_fraction": missing_count / rows if rows else None,
            "role": (
                "identifier"
                if field.name in {
                    "timestamp", "groww_symbol", "expiry_date",
                    "option_type", "strike"
                }
                else "feature"
            ),
        }
    atomic_json(manifest, output_dir / "feature_manifest.json")

    print()
    print("=" * 72)
    print("CLEANING COMPLETE")
    print("=" * 72)
    print("Rows:", f"{rows:,}")
    print("Individual OI available:", report["individual_oi_available_fraction"])
    print("Greeks available:", report["greeks_available_fraction"])
    print("Chain OI available:", report["chain_oi_available_fraction"])
    print("True 1-minute continuity:", report["true_1m_continuity_fraction"])
    print("Numeric inf count:", report["numeric_inf_count"])
    print("Clean dataset:", final_path)
    print("Cleaning report:", output_dir / "cleaning_report.json")
    print("Feature manifest:", output_dir / "feature_manifest.json")

if __name__ == "__main__":
    main()
