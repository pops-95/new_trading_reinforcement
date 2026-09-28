#!/usr/bin/env python3
"""
CUDA-assisted quality analysis for BANKNIFTY 5-minute candidates.

Default input
-------------
data/banknifty_5m_candidates/candidate_dataset.parquet

Outputs
-------
data/banknifty_5m_candidate_analysis/
    candidate_quality_summary.json
    momentum_bucket_analysis.csv
    momentum_exhaustion_analysis.csv
    availability_subgroup_analysis.csv
    ce_momentum_hit_rates.png
    pe_momentum_hit_rates.png
    combined_momentum_hit_rates.png

Analysis
--------
- CE, PE, and combined
- momentum buckets:
    0-50, 50-60, 60-70, 70-80, 80-90, 90-100
- eligibility rate
- +20/+30/+40 before -20 hit rates
- MFE/MAE mean & median at 15/30/45/60 min
- mean/median time-to-target bars and time-to-stop bars
- momentum x exhaustion 2D buckets
- top 5/10/20% success rates and lift versus baseline
- correlation of momentum/exhaustion with outcomes
- availability subgroup analysis:
    chain OI
    individual OI
    Greeks
    valid next-bar entry

CUDA
----
CuPy is used for:
- momentum/exhaustion bucket assignment
- means/medians
- correlations
- top-fraction ranking helpers where practical

Memory safety
-------------
Only required columns are loaded from parquet.
For this 5-minute candidate dataset (~tens of thousands of rows), memory usage
should remain modest. PyArrow iter_batches is used to avoid one monolithic read.

Install
-------
pip install pandas numpy pyarrow matplotlib
pip install cupy-cuda12x

Run
---
python3 analyze_5m_candidate_quality_parts.py \
  --input-dir data/banknifty_5m_candidates \
  --output-dir data/banknifty_5m_candidate_analysis
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import cupy as cp
except Exception as exc:
    raise SystemExit(
        "CuPy is required. Install cupy-cuda12x (or matching CUDA build).\n"
        f"Import error: {exc}"
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MOMENTUM_BINS = [0, 50, 60, 70, 80, 90, 100.000001]
MOMENTUM_LABELS = ["0-50", "50-60", "60-70", "70-80", "80-90", "90-100"]

EXHAUSTION_BINS = [0, 25, 50, 75, 100.000001]
EXHAUSTION_LABELS = ["0-25", "25-50", "50-75", "75-100"]

TARGETS = (20, 30, 40)
HORIZONS = (15, 30, 45, 60)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def release_memory() -> None:
    gc.collect()
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def cuda_info() -> Dict[str, Any]:
    d = cp.cuda.Device()
    p = cp.cuda.runtime.getDeviceProperties(d.id)
    free_b, total_b = cp.cuda.runtime.memGetInfo()

    name = p.get("name", b"unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="ignore")

    return {
        "device_id": int(d.id),
        "name": str(name),
        "compute_capability": f"{int(p.get('major',0))}.{int(p.get('minor',0))}",
        "free_gb": round(free_b / 1024**3, 3),
        "total_gb": round(total_b / 1024**3, 3),
    }


def safe_mean_gpu(a: np.ndarray) -> float:
    if len(a) == 0:
        return np.nan

    x = cp.asarray(a, dtype=cp.float32)
    valid = cp.isfinite(x)

    if int(valid.sum()) == 0:
        del x, valid
        return np.nan

    out = float(cp.asnumpy(cp.mean(x[valid])))
    del x, valid
    return out


def safe_median_gpu(a: np.ndarray) -> float:
    if len(a) == 0:
        return np.nan

    x = cp.asarray(a, dtype=cp.float32)
    valid = cp.isfinite(x)

    if int(valid.sum()) == 0:
        del x, valid
        return np.nan

    out = float(cp.asnumpy(cp.median(x[valid])))
    del x, valid
    return out


def corr_gpu(a: np.ndarray, b: np.ndarray) -> float:
    x = cp.asarray(a, dtype=cp.float32)
    y = cp.asarray(b, dtype=cp.float32)

    valid = cp.isfinite(x) & cp.isfinite(y)

    if int(valid.sum()) < 3:
        del x, y, valid
        return np.nan

    xx = x[valid]
    yy = y[valid]

    if float(cp.std(xx)) == 0 or float(cp.std(yy)) == 0:
        del x, y, valid, xx, yy
        return np.nan

    r = float(cp.asnumpy(cp.corrcoef(xx, yy)[0, 1]))

    del x, y, valid, xx, yy
    return r


def bucket_gpu(values: np.ndarray, bins: Sequence[float]) -> np.ndarray:
    x = cp.asarray(values, dtype=cp.float32)
    b = cp.asarray(bins, dtype=cp.float32)

    idx = cp.digitize(
        x,
        b[1:-1],
        right=False,
    )

    idx = cp.where(
        cp.isfinite(x),
        idx,
        -1,
    )

    out = cp.asnumpy(idx).astype(np.int16)

    del x, b, idx
    return out


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    return obj


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

def side_columns(side: str) -> Dict[str, str]:
    p = side.lower()

    return {
        "momentum": f"{p}_momentum_score",
        "exhaustion": f"{p}_exhaustion_score",
        "eligible": f"{p}_momentum_eligible",

        "individual_oi": f"{p}_individual_oi_available_5m",
        "greeks": f"{p}_greeks_available_5m",

        "entry_valid": f"label_{p}_entry_valid",

        "hit20": f"label_{p}_hit_20_before_stop",
        "hit30": f"label_{p}_hit_30_before_stop",
        "hit40": f"label_{p}_hit_40_before_stop",

        "mfe15": f"label_{p}_mfe_15m",
        "mfe30": f"label_{p}_mfe_30m",
        "mfe45": f"label_{p}_mfe_45m",
        "mfe60": f"label_{p}_mfe_60m",

        "mae15": f"label_{p}_mae_15m",
        "mae30": f"label_{p}_mae_30m",
        "mae45": f"label_{p}_mae_45m",
        "mae60": f"label_{p}_mae_60m",

        "time20": f"label_{p}_time_to_20_bars",
        "time30": f"label_{p}_time_to_30_bars",
        "time40": f"label_{p}_time_to_40_bars",
        "time_stop": f"label_{p}_time_to_stop_bars",
    }


# ---------------------------------------------------------------------------
# Low-memory loading
# ---------------------------------------------------------------------------

def resolve_parquet_inputs(
    input_file: str | None,
    input_dir: str | None,
) -> List[Path]:
    """
    Resolve either:
      - a single parquet file
      - a directory containing candidate_parts/day_*.parquet
      - a direct candidate_parts directory
    """
    paths: List[Path] = []

    if input_file:
        p = Path(input_file)
        if p.exists() and p.is_file():
            paths = [p]
        elif "*" in str(input_file):
            paths = sorted(Path().glob(str(input_file)))
        else:
            raise SystemExit(f"Input file not found: {p}")

    elif input_dir:
        d = Path(input_dir)

        if not d.exists():
            raise SystemExit(f"Input directory not found: {d}")

        # Accept either .../banknifty_5m_candidates or .../candidate_parts
        candidate_parts = d / "candidate_parts"

        if candidate_parts.exists():
            d = candidate_parts

        paths = sorted(d.glob("day_*.parquet"))

    if not paths:
        raise SystemExit(
            "No candidate parquet files found. "
            "Expected candidate_parts/day_*.parquet"
        )

    # Skip unreadable / zero-column parquet files defensively.
    valid = []

    for p in paths:
        try:
            pf = pq.ParquetFile(p)

            if (
                len(pf.schema.names) > 0
                and pf.metadata is not None
                and pf.metadata.num_rows > 0
            ):
                valid.append(p)

        except Exception:
            continue

    if not valid:
        raise SystemExit("No valid candidate parquet parts found.")

    return valid


def load_side(
    parquet_files: Sequence[Path],
    side: str,
    batch_rows: int,
) -> pd.DataFrame:

    c = side_columns(side)

    critical = [
        c["momentum"],
        c["hit20"],
        c["hit30"],
        c["hit40"],
    ]

    chunks = []

    for file_path in parquet_files:

        pf = pq.ParquetFile(file_path)
        available = set(pf.schema.names)

        missing = [
            col for col in critical
            if col not in available
        ]

        if missing:
            # Skip incompatible/old part instead of crashing the whole analysis.
            print(
                f"Skipping {file_path.name}: "
                f"missing required columns {missing}"
            )
            continue

        wanted = list(c.values()) + [
            "chain_oi_available",
        ]

        wanted = [
            col for col in wanted
            if col in available
        ]

        for rb in pf.iter_batches(
            batch_size=batch_rows,
            columns=wanted,
        ):
            chunks.append(
                rb.to_pandas()
            )

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    del chunks
    gc.collect()

    return df


# ---------------------------------------------------------------------------
# Core summary block
# ---------------------------------------------------------------------------

def summarize_subset(
    df: pd.DataFrame,
    side: str,
) -> Dict[str, Any]:

    c = side_columns(side)

    result: Dict[str, Any] = {
        "count": int(len(df))
    }

    for target in TARGETS:
        col = c[f"hit{target}"]

        if col in df:
            result[f"hit_{target}_before_stop_rate"] = safe_mean_gpu(
                df[col].to_numpy(float)
            )
        else:
            result[f"hit_{target}_before_stop_rate"] = np.nan

    for horizon in HORIZONS:

        mfe_col = c[f"mfe{horizon}"]
        mae_col = c[f"mae{horizon}"]

        mfe = (
            df[mfe_col].to_numpy(float)
            if mfe_col in df
            else np.array([])
        )

        mae = (
            df[mae_col].to_numpy(float)
            if mae_col in df
            else np.array([])
        )

        result[f"mfe_{horizon}m_mean"] = safe_mean_gpu(mfe)
        result[f"mfe_{horizon}m_median"] = safe_median_gpu(mfe)

        result[f"mae_{horizon}m_mean"] = safe_mean_gpu(mae)
        result[f"mae_{horizon}m_median"] = safe_median_gpu(mae)

    for target in TARGETS:
        col = c[f"time{target}"]

        arr = (
            df[col].to_numpy(float)
            if col in df
            else np.array([])
        )

        result[f"time_to_{target}_bars_mean"] = safe_mean_gpu(arr)
        result[f"time_to_{target}_bars_median"] = safe_median_gpu(arr)

    stop_col = c["time_stop"]

    stop = (
        df[stop_col].to_numpy(float)
        if stop_col in df
        else np.array([])
    )

    result["time_to_stop_bars_mean"] = safe_mean_gpu(stop)
    result["time_to_stop_bars_median"] = safe_median_gpu(stop)

    elig_col = c["eligible"]

    result["eligibility_rate"] = (
        safe_mean_gpu(
            df[elig_col].to_numpy(float)
        )
        if elig_col in df
        else np.nan
    )

    entry_col = c["entry_valid"]

    result["entry_valid_rate"] = (
        safe_mean_gpu(
            df[entry_col].to_numpy(float)
        )
        if entry_col in df
        else np.nan
    )

    return result


# ---------------------------------------------------------------------------
# Momentum bucket analysis
# ---------------------------------------------------------------------------

def momentum_bucket_analysis(
    df: pd.DataFrame,
    side: str,
) -> pd.DataFrame:

    c = side_columns(side)

    idx = bucket_gpu(
        df[c["momentum"]].to_numpy(float),
        MOMENTUM_BINS,
    )

    baseline = summarize_subset(
        df,
        side,
    )

    rows = []

    for i, label in enumerate(MOMENTUM_LABELS):

        sub = df[idx == i]

        s = summarize_subset(
            sub,
            side,
        )

        s["side"] = side.upper()
        s["momentum_bucket"] = label

        for target in TARGETS:

            rate = s[
                f"hit_{target}_before_stop_rate"
            ]

            base = baseline[
                f"hit_{target}_before_stop_rate"
            ]

            s[
                f"hit_{target}_lift_vs_baseline"
            ] = (
                rate / base
                if pd.notna(rate)
                and pd.notna(base)
                and base != 0
                else np.nan
            )

        rows.append(s)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Momentum x exhaustion
# ---------------------------------------------------------------------------

def momentum_exhaustion_analysis(
    df: pd.DataFrame,
    side: str,
) -> pd.DataFrame:

    c = side_columns(side)

    if c["exhaustion"] not in df:
        return pd.DataFrame()

    mi = bucket_gpu(
        df[c["momentum"]].to_numpy(float),
        MOMENTUM_BINS,
    )

    ei = bucket_gpu(
        df[c["exhaustion"]].to_numpy(float),
        EXHAUSTION_BINS,
    )

    rows = []

    for i, mlab in enumerate(MOMENTUM_LABELS):
        for j, elab in enumerate(EXHAUSTION_LABELS):

            sub = df[
                (mi == i)
                & (ei == j)
            ]

            s = summarize_subset(
                sub,
                side,
            )

            s["side"] = side.upper()
            s["momentum_bucket"] = mlab
            s["exhaustion_bucket"] = elab

            rows.append(s)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Availability subgroup analysis
# ---------------------------------------------------------------------------

def availability_analysis(
    df: pd.DataFrame,
    side: str,
) -> pd.DataFrame:

    c = side_columns(side)

    groups: List[tuple[str, str]] = []

    if "chain_oi_available" in df:
        groups.append(
            ("chain_oi", "chain_oi_available")
        )

    if c["individual_oi"] in df:
        groups.append(
            ("individual_oi", c["individual_oi"])
        )

    if c["greeks"] in df:
        groups.append(
            ("greeks", c["greeks"])
        )

    if c["entry_valid"] in df:
        groups.append(
            ("entry_valid", c["entry_valid"])
        )

    rows = []

    for group_name, col in groups:

        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        for flag in (0, 1):

            sub = df[
                values.fillna(0).astype(int) == flag
            ]

            s = summarize_subset(
                sub,
                side,
            )

            s["side"] = side.upper()
            s["availability_group"] = group_name
            s["available"] = flag

            rows.append(s)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top fraction & separability summary
# ---------------------------------------------------------------------------

def top_fraction_analysis(
    df: pd.DataFrame,
    side: str,
    fractions=(0.05, 0.10, 0.20),
) -> Dict[str, Any]:

    c = side_columns(side)

    d = df[
        [
            c["momentum"],
            c["hit20"],
            c["hit30"],
            c["hit40"],
        ]
    ].copy()

    d = d.rename(
        columns={
            c["momentum"]: "momentum",
            c["hit20"]: "hit20",
            c["hit30"]: "hit30",
            c["hit40"]: "hit40",
        }
    )

    d = d[
        d["momentum"].notna()
    ].sort_values(
        "momentum",
        ascending=False,
    )

    result: Dict[str, Any] = {}

    if d.empty:
        return result

    baselines = {
        t: safe_mean_gpu(
            d[f"hit{t}"].to_numpy(float)
        )
        for t in TARGETS
    }

    for frac in fractions:

        n = max(
            1,
            int(np.ceil(len(d) * frac))
        )

        sub = d.iloc[:n]

        block: Dict[str, Any] = {
            "rows": int(n),
            "momentum_min": float(
                sub["momentum"].min()
            ),
            "momentum_mean": float(
                sub["momentum"].mean()
            ),
        }

        for target in TARGETS:

            rate = safe_mean_gpu(
                sub[f"hit{target}"].to_numpy(float)
            )

            base = baselines[target]

            block[f"hit{target}_rate"] = rate
            block[f"hit{target}_lift"] = (
                rate / base
                if pd.notna(rate)
                and pd.notna(base)
                and base != 0
                else np.nan
            )

        result[
            f"top_{int(frac*100)}pct"
        ] = block

    return result


def separability_summary(
    df: pd.DataFrame,
    side: str,
) -> Dict[str, Any]:

    c = side_columns(side)

    momentum = df[
        c["momentum"]
    ].to_numpy(float)

    out: Dict[str, Any] = {
        "rows": int(len(df)),
        "baseline": summarize_subset(
            df,
            side,
        ),
        "top_fraction": top_fraction_analysis(
            df,
            side,
        ),
    }

    for target in TARGETS:

        y = df[
            c[f"hit{target}"]
        ].to_numpy(float)

        out[
            f"corr_momentum_vs_hit{target}"
        ] = corr_gpu(
            momentum,
            y,
        )

    if c["exhaustion"] in df:

        exhaustion = df[
            c["exhaustion"]
        ].to_numpy(float)

        for target in TARGETS:

            y = df[
                c[f"hit{target}"]
            ].to_numpy(float)

            out[
                f"corr_exhaustion_vs_hit{target}"
            ] = corr_gpu(
                exhaustion,
                y,
            )

    # Compact non-trading diagnostic flag.
    lifts = []

    top10 = out[
        "top_fraction"
    ].get(
        "top_10pct",
        {}
    )

    for target in TARGETS:
        v = top10.get(
            f"hit{target}_lift"
        )
        if v is not None and np.isfinite(v):
            lifts.append(v)

    if not lifts:
        out["separability_flag"] = "INSUFFICIENT_DATA"
    elif max(lifts) >= 1.15:
        out["separability_flag"] = "MATERIAL_LIFT_PRESENT"
    elif max(lifts) >= 1.05:
        out["separability_flag"] = "WEAK_LIFT_PRESENT"
    else:
        out["separability_flag"] = "LITTLE_OR_NO_LIFT"

    return out


# ---------------------------------------------------------------------------
# Combined analysis
# ---------------------------------------------------------------------------

def normalize_side_for_combined(
    df: pd.DataFrame,
    side: str,
) -> pd.DataFrame:

    c = side_columns(side)

    out = pd.DataFrame()

    out["momentum"] = df[c["momentum"]]

    out["eligible"] = (
        df[c["eligible"]]
        if c["eligible"] in df
        else np.nan
    )

    for target in TARGETS:
        out[f"hit{target}"] = df[
            c[f"hit{target}"]
        ]

    for h in HORIZONS:

        out[f"mfe{h}"] = (
            df[c[f"mfe{h}"]]
            if c[f"mfe{h}"] in df
            else np.nan
        )

        out[f"mae{h}"] = (
            df[c[f"mae{h}"]]
            if c[f"mae{h}"] in df
            else np.nan
        )

    for target in TARGETS:
        out[f"time{target}"] = (
            df[c[f"time{target}"]]
            if c[f"time{target}"] in df
            else np.nan
        )

    out["time_stop"] = (
        df[c["time_stop"]]
        if c["time_stop"] in df
        else np.nan
    )

    return out


def combined_momentum_table(
    ce: pd.DataFrame,
    pe: pd.DataFrame,
) -> pd.DataFrame:

    d = pd.concat(
        [
            normalize_side_for_combined(
                ce,
                "ce",
            ),
            normalize_side_for_combined(
                pe,
                "pe",
            ),
        ],
        ignore_index=True,
    )

    idx = bucket_gpu(
        d["momentum"].to_numpy(float),
        MOMENTUM_BINS,
    )

    rows = []

    for i, label in enumerate(MOMENTUM_LABELS):

        sub = d[idx == i]

        row: Dict[str, Any] = {
            "side": "COMBINED",
            "momentum_bucket": label,
            "count": int(len(sub)),
            "eligibility_rate": safe_mean_gpu(
                sub["eligible"].to_numpy(float)
            ),
        }

        for target in TARGETS:

            row[
                f"hit_{target}_before_stop_rate"
            ] = safe_mean_gpu(
                sub[
                    f"hit{target}"
                ].to_numpy(float)
            )

        for h in HORIZONS:

            mfe = sub[
                f"mfe{h}"
            ].to_numpy(float)

            mae = sub[
                f"mae{h}"
            ].to_numpy(float)

            row[
                f"mfe_{h}m_mean"
            ] = safe_mean_gpu(mfe)

            row[
                f"mfe_{h}m_median"
            ] = safe_median_gpu(mfe)

            row[
                f"mae_{h}m_mean"
            ] = safe_mean_gpu(mae)

            row[
                f"mae_{h}m_median"
            ] = safe_median_gpu(mae)

        for target in TARGETS:

            arr = sub[
                f"time{target}"
            ].to_numpy(float)

            row[
                f"time_to_{target}_bars_mean"
            ] = safe_mean_gpu(arr)

            row[
                f"time_to_{target}_bars_median"
            ] = safe_median_gpu(arr)

        stop = sub[
            "time_stop"
        ].to_numpy(float)

        row[
            "time_to_stop_bars_mean"
        ] = safe_mean_gpu(stop)

        row[
            "time_to_stop_bars_median"
        ] = safe_median_gpu(stop)

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def create_charts(
    bucket_df: pd.DataFrame,
    output_dir: Path,
) -> None:

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipping charts.")
        return

    for side in ("CE", "PE", "COMBINED"):

        sub = bucket_df[
            bucket_df["side"] == side
        ].copy()

        if sub.empty:
            continue

        fig = plt.figure(
            figsize=(10, 6)
        )

        plt.plot(
            sub["momentum_bucket"],
            sub["hit_20_before_stop_rate"],
            marker="o",
            label="+20 before -20",
        )

        plt.plot(
            sub["momentum_bucket"],
            sub["hit_30_before_stop_rate"],
            marker="o",
            label="+30 before -20",
        )

        plt.plot(
            sub["momentum_bucket"],
            sub["hit_40_before_stop_rate"],
            marker="o",
            label="+40 before -20",
        )

        plt.xlabel(
            "5-minute momentum score bucket"
        )

        plt.ylabel(
            "Outcome hit rate"
        )

        plt.title(
            f"{side}: 5-minute momentum bucket vs option outcome"
        )

        plt.legend()
        plt.tight_layout()

        fig.savefig(
            output_dir
            / f"{side.lower()}_momentum_hit_rates.png",
            dpi=150,
        )

        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input-file",
        default=None,
        help="Optional single combined candidate parquet.",
    )

    ap.add_argument(
        "--input-dir",
        default="data/banknifty_5m_candidates",
        help=(
            "Directory containing candidate_parts/day_*.parquet, "
            "or candidate_parts itself."
        ),
    )

    ap.add_argument(
        "--output-dir",
        default=(
            "data/banknifty_5m_candidate_analysis"
        ),
    )

    ap.add_argument(
        "--batch-rows",
        type=int,
        default=50000,
    )

    ap.add_argument(
        "--side",
        choices=["both", "ce", "pe"],
        default="both",
    )

    ap.add_argument(
        "--no-charts",
        action="store_true",
    )

    args = ap.parse_args()

    output_dir = Path(
        args.output_dir
    )

    parquet_files = resolve_parquet_inputs(
        args.input_file,
        args.input_dir,
    )

    print(
        f"Candidate parquet parts: {len(parquet_files):,}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dev = cuda_info()

    print(
        f"CUDA: {dev['name']} | "
        f"CC={dev['compute_capability']} | "
        f"free={dev['free_gb']:.2f}/"
        f"{dev['total_gb']:.2f}GB"
    )

    sides = (
        ["ce", "pe"]
        if args.side == "both"
        else [args.side]
    )

    data: Dict[str, pd.DataFrame] = {}
    bucket_frames = []
    mex_frames = []
    availability_frames = []

    summary: Dict[str, Any] = {
        "cuda": dev,
        "input_source": (
            str(args.input_file)
            if args.input_file
            else str(args.input_dir)
        ),
        "parquet_part_count": int(len(parquet_files)),
        "sides": {},
    }

    for side in sides:

        print(
            f"Loading {side.upper()} columns..."
        )

        df = load_side(
            parquet_files,
            side,
            args.batch_rows,
        )

        data[side] = df

        print(
            f"{side.upper()} rows: {len(df):,}"
        )

        print(
            f"Analyzing {side.upper()} momentum buckets..."
        )

        bucket_frames.append(
            momentum_bucket_analysis(
                df,
                side,
            )
        )

        print(
            f"Analyzing {side.upper()} momentum x exhaustion..."
        )

        mex = momentum_exhaustion_analysis(
            df,
            side,
        )

        if not mex.empty:
            mex_frames.append(
                mex
            )

        print(
            f"Analyzing {side.upper()} availability groups..."
        )

        av = availability_analysis(
            df,
            side,
        )

        if not av.empty:
            availability_frames.append(
                av
            )

        summary[
            "sides"
        ][
            side.upper()
        ] = separability_summary(
            df,
            side,
        )

        release_memory()

    if (
        "ce" in data
        and "pe" in data
    ):
        print(
            "Building combined momentum analysis..."
        )

        bucket_frames.append(
            combined_momentum_table(
                data["ce"],
                data["pe"],
            )
        )

    bucket_df = pd.concat(
        bucket_frames,
        ignore_index=True,
    )

    bucket_df.to_csv(
        output_dir
        / "momentum_bucket_analysis.csv",
        index=False,
    )

    mex_df = (
        pd.concat(
            mex_frames,
            ignore_index=True,
        )
        if mex_frames
        else pd.DataFrame()
    )

    mex_df.to_csv(
        output_dir
        / "momentum_exhaustion_analysis.csv",
        index=False,
    )

    availability_df = (
        pd.concat(
            availability_frames,
            ignore_index=True,
        )
        if availability_frames
        else pd.DataFrame()
    )

    availability_df.to_csv(
        output_dir
        / "availability_subgroup_analysis.csv",
        index=False,
    )

    with open(
        output_dir
        / "candidate_quality_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_safe(summary),
            f,
            indent=2,
        )

    if not args.no_charts:
        create_charts(
            bucket_df,
            output_dir,
        )

    print()
    print("=" * 72)
    print(
        "5-MINUTE CANDIDATE QUALITY ANALYSIS COMPLETE"
    )
    print("=" * 72)

    print(
        "Summary:",
        output_dir
        / "candidate_quality_summary.json",
    )

    print(
        "Momentum buckets:",
        output_dir
        / "momentum_bucket_analysis.csv",
    )

    print(
        "Momentum x exhaustion:",
        output_dir
        / "momentum_exhaustion_analysis.csv",
    )

    print(
        "Availability:",
        output_dir
        / "availability_subgroup_analysis.csv",
    )


if __name__ == "__main__":
    main()
