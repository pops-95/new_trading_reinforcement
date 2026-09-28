#!/usr/bin/env python3
"""
Low-memory CUDA-first BANKNIFTY 5-minute dataset builder.

Why this version exists
-----------------------
The earlier implementation could make the machine unresponsive because large
DuckDB GROUP BY / ORDER BY and whole-file pandas operations competed for RAM.

This version is deliberately conservative on host memory:
- processes option data ONE TRADING DAY AT A TIME
- never loads the complete option parquet into pandas
- uses DuckDB only for bounded daily extraction/resampling
- uses CuPy/CUDA for feature arithmetic
- carries only a small rolling state per option symbol across days
- writes daily parquet parts immediately
- avoids a global multi-million-row window/sort pass
- combines parts only at the end using DuckDB with a strict memory cap

Default settings are tuned for an 8 GB Quadro P4000 and modest system RAM.

Outputs
-------
data/banknifty_5m/
    banknifty_underlying_5m.parquet
    option_parts/day_YYYYMMDD.parquet
    banknifty_options_5m_features.parquet
    banknifty_5m_quality_report.json
    banknifty_5m_feature_manifest.json

Timing convention
-----------------
The 09:15..09:19 source minutes form one completed bar timestamped 09:20.
Decisions can therefore use that bar only at/after 09:20.

No forward fill. No backward fill. No future leakage.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb
import cupy as cp


# ---------------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------------

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


def open_duckdb(threads: int, memory_gb: float, temp_dir: Path):
    temp_dir.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()
    c.execute(f"PRAGMA threads={max(1, int(threads))}")
    c.execute("PRAGMA preserve_insertion_order=false")
    c.execute(f"PRAGMA memory_limit='{float(memory_gb):.2f}GB'")
    td = str(temp_dir).replace("'", "''")
    c.execute(f"PRAGMA temp_directory='{td}'")
    return c


def parquet_columns(path: Path) -> List[str]:
    return pq.ParquetFile(path).schema.names


def sqlq(path: Path) -> str:
    return str(path).replace("'", "''")


def bucket_sql(ts: str = "timestamp") -> str:
    return f"""(
      date_trunc('day',{ts}) + INTERVAL '9 hours 15 minutes'
      + (FLOOR(
           date_diff(
             'minute',
             date_trunc('day',{ts}) + INTERVAL '9 hours 15 minutes',
             {ts}
           ) / 5.0
         ) * 5) * INTERVAL '1 minute'
    )"""


# ---------------------------------------------------------------------------
# CUDA vector helpers
# ---------------------------------------------------------------------------

def gpu_clean(a, positive=False, nonnegative=False):
    x = cp.asarray(a, dtype=cp.float32)
    x = cp.where(cp.isfinite(x), x, cp.nan)
    if positive:
        x = cp.where(x > 0, x, cp.nan)
    elif nonnegative:
        x = cp.where(x >= 0, x, cp.nan)
    out = cp.asnumpy(x)
    del x
    return out


def gpu_diff(cur, prev, valid):
    c = cp.asarray(cur, dtype=cp.float32)
    p = cp.asarray(prev, dtype=cp.float32)
    m = cp.asarray(valid, dtype=cp.bool_)
    out = cp.where(m & cp.isfinite(c) & cp.isfinite(p), c - p, cp.nan)
    r = cp.asnumpy(out)
    del c, p, m, out
    return r


def gpu_pct(cur, prev, valid):
    c = cp.asarray(cur, dtype=cp.float32)
    p = cp.asarray(prev, dtype=cp.float32)
    m = cp.asarray(valid, dtype=cp.bool_)
    out = cp.where(
        m & cp.isfinite(c) & cp.isfinite(p) & (p != 0),
        (c - p) / p,
        cp.nan,
    )
    r = cp.asnumpy(out)
    del c, p, m, out
    return r


def gpu_ratio(a, b):
    aa = cp.asarray(a, dtype=cp.float32)
    bb = cp.asarray(b, dtype=cp.float32)
    out = cp.where(
        cp.isfinite(aa) & cp.isfinite(bb) & (bb != 0),
        aa / bb,
        cp.nan,
    )
    r = cp.asnumpy(out)
    del aa, bb, out
    return r


def gpu_mul(a, b):
    aa = cp.asarray(a, dtype=cp.float32)
    bb = cp.asarray(b, dtype=cp.float32)
    out = cp.where(cp.isfinite(aa) & cp.isfinite(bb), aa * bb, cp.nan)
    r = cp.asnumpy(out)
    del aa, bb, out
    return r


# ---------------------------------------------------------------------------
# Underlying 5m - small enough to process in one bounded table
# ---------------------------------------------------------------------------

def build_underlying_5m(
    src: Path,
    dst: Path,
    threads: int,
    memory_gb: float,
    temp_dir: Path,
) -> None:
    available = set(parquet_columns(src))
    needed = {"timestamp", "open", "high", "low", "close"}
    missing = needed - available
    if missing:
        raise RuntimeError(f"Underlying missing: {sorted(missing)}")

    vol = "SUM(volume) AS volume" if "volume" in available else "NULL::DOUBLE AS volume"
    s = sqlq(src)
    d = sqlq(dst)
    b = bucket_sql()

    c = open_duckdb(threads, memory_gb, temp_dir)
    c.execute(f"""
    COPY (
      WITH x AS (
        SELECT *, {b} AS bar_start
        FROM read_parquet('{s}')
        WHERE timestamp IS NOT NULL
          AND CAST(timestamp AS TIME) >= TIME '09:15:00'
          AND CAST(timestamp AS TIME) <  TIME '15:30:00'
      )
      SELECT
        bar_start,
        bar_start + INTERVAL '5 minutes' AS timestamp,
        ARG_MIN(open,timestamp)  AS open,
        MAX(high)                AS high,
        MIN(low)                 AS low,
        ARG_MAX(close,timestamp) AS close,
        {vol},
        COUNT(close)             AS source_1m_rows
      FROM x
      GROUP BY bar_start
      ORDER BY bar_start
    ) TO '{d}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    c.close()

    # Underlying 5m is only ~50k rows, so this stays small.
    x = pd.read_parquet(dst)
    x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")

    for col in ("open", "high", "low", "close"):
        x[col] = gpu_clean(pd.to_numeric(x[col], errors="coerce"), positive=True)

    if "volume" in x:
        x["volume"] = gpu_clean(pd.to_numeric(x["volume"], errors="coerce"), nonnegative=True)

    close = x["close"]
    high = x["high"]
    low = x["low"]
    open_ = x["open"]

    for mins, bars in ((5,1),(10,2),(15,3),(30,6)):
        x[f"return_{mins}m"] = close.pct_change(bars)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high-low, (high-prev_close).abs(), (low-prev_close).abs()],
        axis=1,
    ).max(axis=1)

    x["true_range_5m"] = tr
    x["atr14_5m"] = tr.rolling(14, min_periods=14).mean()
    x["atr5_5m"] = tr.rolling(5, min_periods=5).mean()
    x["atr20_5m"] = tr.rolling(20, min_periods=20).mean()
    x["atr_expansion_5m"] = x["atr5_5m"] / x["atr20_5m"].replace(0, np.nan)

    rng = (high-low).replace(0, np.nan)
    x["body_to_range_5m"] = (close-open_).abs() / rng
    x["close_location_5m"] = (close-low) / rng

    for span in (9,20,50):
        ema = close.ewm(span=span, adjust=False).mean()
        x[f"ema{span}_5m"] = ema
        x[f"ema{span}_distance_atr"] = (close-ema) / x["atr14_5m"].replace(0, np.nan)
        x[f"ema{span}_slope_3bars"] = ema.diff(3)

    x["velocity_5m"] = close.diff() / x["atr14_5m"].replace(0, np.nan)
    x["velocity_10m"] = close.diff(2) / x["atr14_5m"].replace(0, np.nan)
    x["acceleration_5m"] = x["velocity_5m"].diff()
    x["jerk_5m"] = x["acceleration_5m"].diff()

    for bars in (3,6):
        path = close.diff().abs().rolling(bars, min_periods=bars).sum()
        x[f"path_efficiency_{bars}bars"] = (
            close.diff(bars).abs() / path.replace(0, np.nan)
        ).clip(0,1)

    x["date"] = x["timestamp"].dt.normalize()
    g = x.groupby("date", sort=False)
    x["day_open_5m"] = g["open"].transform("first")
    x["day_high_5m"] = g["high"].cummax()
    x["day_low_5m"] = g["low"].cummin()
    x["return_from_day_open"] = (
        (close-x["day_open_5m"]) / x["day_open_5m"].replace(0,np.nan)
    )

    sign = np.sign(close.diff()).fillna(0)
    gr = sign.ne(sign.shift()).cumsum()
    x["direction_sign_5m"] = sign
    x["direction_run_5m"] = np.where(
        sign == 0,
        0,
        sign.groupby(gr).cumcount()+1
    )

    # completed 15m regime
    x["session_bar_index"] = g.cumcount()
    x["block15"] = (x["session_bar_index"] // 3).astype(np.int32)

    f = (
        x.groupby(["date","block15"], sort=False)
        .agg(
            timestamp=("timestamp","max"),
            open=("open","first"),
            high=("high","max"),
            low=("low","min"),
            close=("close","last"),
            count=("close","count"),
        )
        .reset_index(drop=True)
    )

    f = f[f["count"] == 3].copy()
    f["ema_fast_15m"] = f["close"].ewm(span=2, adjust=False).mean()
    f["ema_slow_15m"] = f["close"].ewm(span=4, adjust=False).mean()
    f["return_15m_regime"] = f["close"].pct_change()
    f["ema_spread_15m"] = (
        (f["ema_fast_15m"]-f["ema_slow_15m"])
        / f["ema_slow_15m"].replace(0,np.nan)
    )
    f["direction_15m"] = np.select(
        [
            (f["ema_fast_15m"] > f["ema_slow_15m"]) & (f["return_15m_regime"] > 0),
            (f["ema_fast_15m"] < f["ema_slow_15m"]) & (f["return_15m_regime"] < 0),
        ],
        [1.0,-1.0],
        default=0.0,
    )

    x = pd.merge_asof(
        x.sort_values("timestamp"),
        f[["timestamp","return_15m_regime","ema_spread_15m","direction_15m"]]
        .sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    x["regime_15m_available"] = x["direction_15m"].notna().astype("int8")

    atomic_parquet(x.replace([np.inf,-np.inf],np.nan), dst)
    del x, f
    release_memory()


# ---------------------------------------------------------------------------
# trading-day discovery
# ---------------------------------------------------------------------------

def get_trading_days(
    option_file: Path,
    threads: int,
    memory_gb: float,
    temp_dir: Path,
) -> List[pd.Timestamp]:
    c = open_duckdb(threads, memory_gb, temp_dir)
    s = sqlq(option_file)
    d = c.execute(f"""
      SELECT DISTINCT CAST(timestamp AS DATE) AS d
      FROM read_parquet('{s}')
      WHERE timestamp IS NOT NULL
        AND CAST(timestamp AS TIME) >= TIME '09:15:00'
        AND CAST(timestamp AS TIME) <  TIME '15:30:00'
      ORDER BY d
    """).df()
    c.close()
    return [pd.Timestamp(v) for v in d["d"].tolist()]


# ---------------------------------------------------------------------------
# one-day bounded resampling
# ---------------------------------------------------------------------------

def day_resample_sql(
    option_file: Path,
    day: pd.Timestamp,
    output_part: Path,
    threads: int,
    memory_gb: float,
    temp_dir: Path,
) -> None:
    available = set(parquet_columns(option_file))
    b = bucket_sql()
    s = sqlq(option_file)
    d = sqlq(output_part)
    ds = day.strftime("%Y-%m-%d")

    vol = "SUM(volume) AS volume" if "volume" in available else "NULL::DOUBLE AS volume"
    oi = (
        "ARG_MAX(open_interest,timestamp) AS open_interest"
        if "open_interest" in available
        else "NULL::DOUBLE AS open_interest"
    )

    keep_last = [
        "spot_close",
        "implied_volatility",
        "delta",
        "gamma",
        "theta_per_day",
        "vega_per_pct",
        "theta_burden",
        "gamma_responsiveness",
        "dte",
        "moneyness",
        "abs_moneyness",
        "pcr_oi",
        "pcr_volume",
        "ce_oi_total",
        "pe_oi_total",
        "ce_volume_total",
        "pe_volume_total",
        "ce_oi_center",
        "pe_oi_center",
        "call_oi_center_change",
        "put_oi_center_change",
        "oi_migration_score",
        "chain_price_oi_bias",
        "chain_oi_available",
    ]

    extra = []
    for col in keep_last:
        if col in available:
            extra.append(f"ARG_MAX({col},timestamp) AS {col}")
        else:
            extra.append(f"NULL::DOUBLE AS {col}")

    c = open_duckdb(threads, memory_gb, temp_dir)
    c.execute(f"""
    COPY (
      WITH x AS (
        SELECT *, {b} AS bar_start
        FROM read_parquet('{s}')
        WHERE CAST(timestamp AS DATE) = DATE '{ds}'
          AND CAST(timestamp AS TIME) >= TIME '09:15:00'
          AND CAST(timestamp AS TIME) <  TIME '15:30:00'
          AND groww_symbol IS NOT NULL
          AND option_type IN ('CE','PE')
      )
      SELECT
        groww_symbol,
        strike,
        expiry_date,
        option_type,
        bar_start,
        bar_start + INTERVAL '5 minutes' AS timestamp,
        ARG_MIN(open,timestamp)  AS open,
        MAX(high)                AS high,
        MIN(low)                 AS low,
        ARG_MAX(close,timestamp) AS close,
        {vol},
        {oi},
        {",".join(extra)},
        COUNT(close) AS source_1m_rows
      FROM x
      GROUP BY
        groww_symbol,
        strike,
        expiry_date,
        option_type,
        bar_start
      ORDER BY groww_symbol,bar_start
    ) TO '{d}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    c.close()


# ---------------------------------------------------------------------------
# compact rolling state
# ---------------------------------------------------------------------------

class SymbolState:
    __slots__ = (
        "ts","close","volume","oi","iv","delta",
        "premium_change","iv_change","oi_change",
    )

    def __init__(self):
        self.ts = deque(maxlen=12)
        self.close = deque(maxlen=12)
        self.volume = deque(maxlen=12)
        self.oi = deque(maxlen=12)
        self.iv = deque(maxlen=12)
        self.delta = deque(maxlen=12)
        self.premium_change = deque(maxlen=12)
        self.iv_change = deque(maxlen=12)
        self.oi_change = deque(maxlen=12)


def finite(v):
    return v is not None and np.isfinite(v)


def prev_exact(state: SymbolState, ts: pd.Timestamp) -> bool:
    if not state.ts:
        return False
    return (pd.Timestamp(ts) - pd.Timestamp(state.ts[-1])) == pd.Timedelta(minutes=5)


# ---------------------------------------------------------------------------
# daily CUDA feature generation
# ---------------------------------------------------------------------------

def process_day_cuda(
    daily_raw: Path,
    output_part: Path,
    states: Dict[str, SymbolState],
    underlying_5m: pd.DataFrame,
) -> Tuple[int, int]:
    df = pd.read_parquet(daily_raw)
    if df.empty:
        atomic_parquet(df, output_part)
        return 0, 0

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")

    for col in ("open","high","low","close","spot_close","strike"):
        if col in df.columns:
            df[col] = gpu_clean(
                pd.to_numeric(df[col], errors="coerce").to_numpy(),
                positive=True,
            )

    for col in ("volume","open_interest"):
        if col in df.columns:
            df[col] = gpu_clean(
                pd.to_numeric(df[col], errors="coerce").to_numpy(),
                nonnegative=True,
            )

    for col in (
        "implied_volatility","delta","gamma","theta_per_day","vega_per_pct",
        "theta_burden","gamma_responsiveness","dte","moneyness","abs_moneyness",
        "pcr_oi","pcr_volume","ce_oi_total","pe_oi_total","ce_volume_total",
        "pe_volume_total","ce_oi_center","pe_oi_center","call_oi_center_change",
        "put_oi_center_change","oi_migration_score","chain_price_oi_bias"
    ):
        if col in df.columns:
            df[col] = gpu_clean(
                pd.to_numeric(df[col], errors="coerce").to_numpy()
            )

    df = df.sort_values(["groww_symbol","timestamp"]).reset_index(drop=True)

    n = len(df)
    prev_close = np.full(n, np.nan, dtype=np.float32)
    prev_vol   = np.full(n, np.nan, dtype=np.float32)
    prev_oi    = np.full(n, np.nan, dtype=np.float32)
    prev_iv    = np.full(n, np.nan, dtype=np.float32)
    prev_delta = np.full(n, np.nan, dtype=np.float32)
    valid_prev = np.zeros(n, dtype=bool)

    ret10 = np.full(n, np.nan, dtype=np.float32)
    ret15 = np.full(n, np.nan, dtype=np.float32)
    vol_acc = np.full(n, np.nan, dtype=np.float32)
    prev_pchg = np.full(n, np.nan, dtype=np.float32)
    prev_ivchg = np.full(n, np.nan, dtype=np.float32)
    prev_oichg = np.full(n, np.nan, dtype=np.float32)

    # Only state lookup/group traversal is CPU; all arithmetic below is CUDA.
    for sym, idxs in df.groupby("groww_symbol", sort=False).groups.items():
        st = states.get(sym)
        if st is None:
            st = SymbolState()
            states[sym] = st

        for idx in idxs:
            ts = df.at[idx, "timestamp"]
            if st.ts and (pd.Timestamp(ts)-pd.Timestamp(st.ts[-1]) == pd.Timedelta(minutes=5)):
                valid_prev[idx] = True
                prev_close[idx] = st.close[-1]
                prev_vol[idx] = st.volume[-1]
                prev_oi[idx] = st.oi[-1]
                prev_iv[idx] = st.iv[-1]
                prev_delta[idx] = st.delta[-1]

                if len(st.close) >= 2 and finite(st.close[-2]) and st.close[-2] != 0:
                    ret10[idx] = (df.at[idx,"close"] - st.close[-2]) / st.close[-2]

                if len(st.close) >= 3 and finite(st.close[-3]) and st.close[-3] != 0:
                    ret15[idx] = (df.at[idx,"close"] - st.close[-3]) / st.close[-3]

                if st.premium_change:
                    prev_pchg[idx] = st.premium_change[-1]
                if st.iv_change:
                    prev_ivchg[idx] = st.iv_change[-1]
                if st.oi_change:
                    prev_oichg[idx] = st.oi_change[-1]

                if len(st.volume) >= 10:
                    v3 = np.array(list(st.volume)[-2:] + [df.at[idx,"volume"]], dtype=float)
                    v10 = np.array(list(st.volume)[-9:] + [df.at[idx,"volume"]], dtype=float)
                    if np.isfinite(v3).any() and np.isfinite(v10).any():
                        m3 = np.nanmean(v3)
                        m10 = np.nanmean(v10)
                        if np.isfinite(m10) and m10 != 0:
                            vol_acc[idx] = m3/m10

            # provisional append happens after CUDA one-step changes are available

    close = df["close"].to_numpy(np.float32)
    volume = df["volume"].to_numpy(np.float32)
    oi = df["open_interest"].to_numpy(np.float32)
    iv = df["implied_volatility"].to_numpy(np.float32)
    delta = df["delta"].to_numpy(np.float32)

    pchg = gpu_diff(close, prev_close, valid_prev)
    oret5 = gpu_pct(close, prev_close, valid_prev)
    vchg = gpu_diff(volume, prev_vol, valid_prev)
    oichg = gpu_diff(oi, prev_oi, valid_prev)
    oipct = gpu_pct(oi, prev_oi, valid_prev)
    ivchg = gpu_diff(iv, prev_iv, valid_prev)
    dchg = gpu_diff(delta, prev_delta, valid_prev)

    df["option_return_5m"] = oret5
    df["option_return_10m"] = ret10
    df["option_return_15m"] = ret15
    df["premium_change_5m"] = pchg
    df["volume_change_5m"] = vchg
    df["volume_acceleration_5m"] = vol_acc
    df["oi_change_5m"] = oichg
    df["oi_pct_change_5m"] = oipct
    df["iv_change_5m"] = ivchg
    df["delta_change_5m"] = dchg

    # CUDA second derivatives
    df["premium_acceleration_5m"] = gpu_diff(pchg, prev_pchg, valid_prev)
    df["iv_acceleration_5m"] = gpu_diff(ivchg, prev_ivchg, valid_prev)
    df["oi_acceleration_5m"] = gpu_diff(oichg, prev_oichg, valid_prev)

    rng = (df["high"]-df["low"]).replace(0,np.nan)
    df["option_body_to_range_5m"] = (df["close"]-df["open"]).abs()/rng
    df["option_close_location_5m"] = (df["close"]-df["low"])/rng
    df["price_oi_change_5m"] = gpu_mul(pchg, oichg)
    df["price_return_x_oi_pct_5m"] = gpu_mul(oret5, oipct)
    df["true_5m_prev_available"] = valid_prev.astype(np.int8)
    df["individual_oi_available_5m"] = np.isfinite(oi).astype(np.int8)

    greek_mask = (
        np.isfinite(iv)
        & np.isfinite(delta)
        & np.isfinite(df["gamma"].to_numpy(float))
        & np.isfinite(df["theta_per_day"].to_numpy(float))
        & np.isfinite(df["vega_per_pct"].to_numpy(float))
    )
    df["greeks_available_5m"] = greek_mask.astype(np.int8)

    # Update compact state now that changes are known.
    for sym, idxs in df.groupby("groww_symbol", sort=False).groups.items():
        st = states[sym]
        for idx in idxs:
            st.ts.append(df.at[idx,"timestamp"])
            st.close.append(float(df.at[idx,"close"]) if pd.notna(df.at[idx,"close"]) else np.nan)
            st.volume.append(float(df.at[idx,"volume"]) if pd.notna(df.at[idx,"volume"]) else np.nan)
            st.oi.append(float(df.at[idx,"open_interest"]) if pd.notna(df.at[idx,"open_interest"]) else np.nan)
            st.iv.append(float(df.at[idx,"implied_volatility"]) if pd.notna(df.at[idx,"implied_volatility"]) else np.nan)
            st.delta.append(float(df.at[idx,"delta"]) if pd.notna(df.at[idx,"delta"]) else np.nan)
            st.premium_change.append(float(pchg[idx]) if np.isfinite(pchg[idx]) else np.nan)
            st.iv_change.append(float(ivchg[idx]) if np.isfinite(ivchg[idx]) else np.nan)
            st.oi_change.append(float(oichg[idx]) if np.isfinite(oichg[idx]) else np.nan)

    # bounded join with the small underlying table
    ctx = underlying_5m[underlying_5m["timestamp"].isin(df["timestamp"].unique())].copy()

    drop_dupes = [c for c in ("open","high","low","close","volume") if c in ctx.columns]
    ctx = ctx.drop(columns=drop_dupes, errors="ignore")

    df = df.merge(ctx, on="timestamp", how="left", suffixes=("","_underlying"))

    df = df.replace([np.inf,-np.inf],np.nan)
    atomic_parquet(df, output_part)

    complete = int((df["source_1m_rows"] == 5).sum())
    rows = len(df)

    del df, ctx
    release_memory()
    return rows, complete


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------

def combine_parts(
    parts_glob: str,
    dst: Path,
    threads: int,
    memory_gb: float,
    temp_dir: Path,
) -> None:
    s = parts_glob.replace("'", "''")
    d = sqlq(dst)

    c = open_duckdb(threads, memory_gb, temp_dir)
    c.execute(f"""
      COPY (
        SELECT *
        FROM read_parquet('{s}')
      )
      TO '{d}'
      (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    c.close()


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------

def quality_report(
    underlying_file: Path,
    option_file: Path,
    outdir: Path,
) -> None:
    s = sqlq(option_file)
    c = duckdb.connect()
    r = c.execute(f"""
      SELECT
        COUNT(*) AS row_count,
        AVG(CAST(source_1m_rows = 5 AS DOUBLE)) AS complete_5m_bar_fraction,
        AVG(CAST(true_5m_prev_available = 1 AS DOUBLE)) AS true_5m_continuity_fraction,
        AVG(CAST(individual_oi_available_5m = 1 AS DOUBLE)) AS individual_oi_available_fraction,
        AVG(CAST(greeks_available_5m = 1 AS DOUBLE)) AS greeks_available_fraction,
        AVG(COALESCE(CAST(chain_oi_available AS DOUBLE),0.0)) AS chain_oi_available_fraction
      FROM read_parquet('{s}')
    """).df().iloc[0].to_dict()
    c.close()

    out = {
        "underlying_5m_rows": int(pq.ParquetFile(underlying_file).metadata.num_rows),
        **{
            k: (
                int(v) if k == "row_count"
                else float(v) if pd.notna(v)
                else None
            )
            for k,v in r.items()
        }
    }

    atomic_json(out, outdir/"banknifty_5m_quality_report.json")


def feature_manifest(option_file: Path, outdir: Path) -> None:
    schema = pq.ParquetFile(option_file).schema_arrow
    ids = {
        "timestamp","bar_start","groww_symbol",
        "expiry_date","option_type","strike"
    }
    m = {}
    for f in schema:
        m[f.name] = {
            "dtype": str(f.type),
            "role": "identifier" if f.name in ids else "feature",
            "causal": True,
        }
    atomic_json(m, outdir/"banknifty_5m_feature_manifest.json")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--option-file",
        default="data/banknifty_cleaned/banknifty_option_features_clean.parquet",
    )
    ap.add_argument(
        "--underlying-file",
        default="data/banknifty_processed_stream/banknifty_underlying_features.parquet",
    )
    ap.add_argument(
        "--output-dir",
        default="data/banknifty_5m",
    )

    # Conservative defaults for P4000 8GB and limited system RAM.
    ap.add_argument("--duckdb-threads", type=int, default=2)
    ap.add_argument("--duckdb-memory-gb", type=float, default=1.5)

    ap.add_argument(
        "--keep-daily-raw",
        action="store_true",
        help="Keep temporary daily resampled files for debugging.",
    )

    ap.add_argument(
        "--no-combine",
        action="store_true",
        help="Leave final data as daily parts; safest for very low RAM systems.",
    )

    ap.add_argument("--fresh", action="store_true")

    args = ap.parse_args()

    option_file = Path(args.option_file)
    underlying_file = Path(args.underlying_file)
    out = Path(args.output_dir)

    if not option_file.exists():
        raise SystemExit(f"Missing option file: {option_file}")
    if not underlying_file.exists():
        raise SystemExit(f"Missing underlying file: {underlying_file}")

    if args.fresh and out.exists():
        shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)
    temp_dir = out/"duckdb_temp"
    raw_dir = out/"daily_raw"
    part_dir = out/"option_parts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)

    dev = cuda_info()
    print(
        f"CUDA: {dev['name']} | CC={dev['compute_capability']} | "
        f"free={dev['free_gb']:.2f}GB / {dev['total_gb']:.2f}GB"
    )
    print(
        f"DuckDB cap: {args.duckdb_memory_gb:.2f} GB | "
        f"threads={args.duckdb_threads}"
    )
    atomic_json(dev, out/"cuda_info.json")

    # ------------------------------------------------------------
    # 1 underlying
    # ------------------------------------------------------------
    underlying_5m = out/"banknifty_underlying_5m.parquet"

    print("1/5 Building underlying 5m context...")
    if not underlying_5m.exists():
        build_underlying_5m(
            underlying_file,
            underlying_5m,
            args.duckdb_threads,
            args.duckdb_memory_gb,
            temp_dir,
        )
    else:
        print("    cached")

    underlying_df = pd.read_parquet(underlying_5m)
    underlying_df["timestamp"] = pd.to_datetime(
        underlying_df["timestamp"], errors="coerce"
    )

    # ------------------------------------------------------------
    # 2 discover days
    # ------------------------------------------------------------
    print("2/5 Discovering trading days...")
    days = get_trading_days(
        option_file,
        args.duckdb_threads,
        args.duckdb_memory_gb,
        temp_dir,
    )
    print(f"    {len(days):,} trading days")

    # ------------------------------------------------------------
    # 3 day-at-a-time option resample + CUDA features
    # ------------------------------------------------------------
    print("3/5 Processing options day-by-day...")

    states: Dict[str, SymbolState] = {}
    total_rows = 0
    total_complete = 0

    for i, day in enumerate(days, 1):
        tag = day.strftime("%Y%m%d")
        raw = raw_dir/f"raw_{tag}.parquet"
        part = part_dir/f"day_{tag}.parquet"

        if part.exists():
            print(
                f"    [{i:04d}/{len(days):04d}] {tag} cached",
                flush=True,
            )
            # IMPORTANT: resume cannot reconstruct long rolling state cheaply.
            # For exact continuity, use --fresh after an interrupted run.
            continue

        print(
            f"    [{i:04d}/{len(days):04d}] {tag}",
            flush=True,
        )

        day_resample_sql(
            option_file,
            day,
            raw,
            args.duckdb_threads,
            args.duckdb_memory_gb,
            temp_dir,
        )

        rows, complete = process_day_cuda(
            raw,
            part,
            states,
            underlying_df,
        )

        total_rows += rows
        total_complete += complete

        if not args.keep_daily_raw:
            try:
                raw.unlink()
            except FileNotFoundError:
                pass

        release_memory()

    del underlying_df
    release_memory()

    # ------------------------------------------------------------
    # 4 combine or keep partitioned
    # ------------------------------------------------------------
    final = out/"banknifty_options_5m_features.parquet"

    print("4/5 Finalizing output...")

    if args.no_combine:
        print(
            "    --no-combine enabled: daily parts are the final dataset."
        )
        final_for_report = None
    else:
        if not final.exists():
            combine_parts(
                str(part_dir/"day_*.parquet"),
                final,
                args.duckdb_threads,
                args.duckdb_memory_gb,
                temp_dir,
            )
        else:
            print("    combined file cached")
        final_for_report = final

    # ------------------------------------------------------------
    # 5 report
    # ------------------------------------------------------------
    print("5/5 Writing report/manifest...")

    if final_for_report is not None:
        quality_report(
            underlying_5m,
            final_for_report,
            out,
        )
        feature_manifest(
            final_for_report,
            out,
        )
    else:
        # manifest from first part, report from partitioned scan
        first = next(part_dir.glob("day_*.parquet"), None)
        if first:
            feature_manifest(first, out)

        # DuckDB can read the partition glob within strict cap.
        tmp_final = out/"_report_only.parquet"
        combine_parts(
            str(part_dir/"day_*.parquet"),
            tmp_final,
            1,
            min(args.duckdb_memory_gb, 1.0),
            temp_dir,
        )
        quality_report(underlying_5m, tmp_final, out)
        try:
            tmp_final.unlink()
        except FileNotFoundError:
            pass

    print()
    print("="*72)
    print("LOW-MEMORY CUDA 5-MINUTE DATASET COMPLETE")
    print("="*72)
    print("Underlying:", underlying_5m)

    if final_for_report is not None:
        print("Options:", final_for_report)
    else:
        print("Options parts:", part_dir)

    print("Quality:", out/"banknifty_5m_quality_report.json")


if __name__ == "__main__":
    main()
