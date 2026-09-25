#!/usr/bin/env python3
"""
Memory-safe streaming CUDA preprocessing for BANKNIFTY option data.

Designed to avoid Linux OOM kills by NEVER loading the full option dataset
into RAM.

Pipeline
--------
1. Load BANKNIFTY underlying once.
2. Build causal 1m / completed 5m / completed 15m features.
3. Process option contract parquet files in small batches:
      - clean raw values
      - compute per-contract option dynamics
      - merge spot/HTF context
      - calculate IV + Greeks on CUDA in row batches
      - write stage-1 parquet part
      - free CPU/GPU memory
4. Use DuckDB directly on stage-1 parquet files to calculate:
      - CE/PE total OI
      - CE/PE total volume
      - OI centres
      - PCR OI / PCR volume
      - OI migration
      - chain price×OI pressure
5. Join chain features back to option rows with DuckDB.
6. Stream the merged parquet in record batches:
      - compute side-relative features
      - momentum maturity score
      - exhaustion score
      - availability flags
      - critical row filtering
      - write final parquet parts
7. Combine final parts into one parquet with DuckDB.
8. Write quality_report.json.

Missing-data rules
------------------
- OHLC <= 0                -> NaN
- volume < 0               -> NaN
- OI < 0                   -> NaN
- genuine volume == 0      -> preserved
- genuine OI == 0          -> preserved
- failed IV solution       -> NaN
- divide by zero           -> NaN
- +/- infinity             -> NaN
- forward fill             -> NEVER
- backward fill            -> NEVER

Recommended for Quadro P4000 8 GB
---------------------------------
Start with:
    --files-per-batch 4
    --gpu-batch-rows 100000
    --final-batch-rows 150000
    --duckdb-threads 4

If RAM is still tight:
    --files-per-batch 2
    --gpu-batch-rows 50000
    --final-batch-rows 75000

Install
-------
pip install pandas numpy pyarrow duckdb

For CUDA 12.x:
pip install cupy-cuda12x

For CUDA 11.x:
pip install cupy-cuda11x

Run
---
python3 preprocess_banknifty_streaming_cuda.py \
    --input-dir data/banknifty_groww \
    --output-dir data/banknifty_processed_stream \
    --files-per-batch 4 \
    --gpu-batch-rows 100000 \
    --final-batch-rows 150000 \
    --duckdb-threads 4
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Limit aggressive CPU threading.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import cupy as cp
    from cupyx.scipy.special import erf
except Exception as exc:
    raise SystemExit(
        "CuPy is required for CUDA preprocessing.\n"
        "CUDA 12.x: pip install cupy-cuda12x\n"
        "CUDA 11.x: pip install cupy-cuda11x\n"
        f"Import error: {exc}"
    )

try:
    import duckdb
except Exception as exc:
    raise SystemExit(
        "DuckDB is required for memory-safe parquet aggregation.\n"
        "Install: pip install duckdb\n"
        f"Import error: {exc}"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RATE = 0.065
DEFAULT_Q = 0.0
DEFAULT_CHAIN_WINGS = 8
DEFAULT_STRIKE_STEP = 100.0

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Generic helpers
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


def safe_div(a, b):
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = aa / bb
    out[~np.isfinite(out)] = np.nan
    if isinstance(a, pd.Series):
        return pd.Series(out, index=a.index)
    return out


def clean_market_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in ("open", "high", "low", "close"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out.loc[out[c] <= 0, c] = np.nan

    for c in ("volume", "open_interest", "strike"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "volume" in out.columns:
        out.loc[out["volume"] < 0, "volume"] = np.nan

    if "open_interest" in out.columns:
        out.loc[out["open_interest"] < 0, "open_interest"] = np.nan

    return out.replace([np.inf, -np.inf], np.nan)


def add_available_flag(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    cols = [c for c in cols if c in df.columns]
    if cols:
        df[name] = df[cols].notna().all(axis=1).astype("int8")
    else:
        df[name] = np.int8(0)


def cuda_device_info() -> Dict[str, Any]:
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)

    name = props.get("name", b"unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="ignore")

    free_b, total_b = cp.cuda.runtime.memGetInfo()

    return {
        "device_id": int(dev.id),
        "name": str(name),
        "free_gb": round(free_b / (1024**3), 3),
        "total_gb": round(total_b / (1024**3), 3),
    }


def release_memory() -> None:
    gc.collect()
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Underlying features
# ---------------------------------------------------------------------------

def load_spot(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Underlying file missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = clean_market_values(df)

    df = (
        df[df["timestamp"].notna()]
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )

    df["date"] = df["timestamp"].dt.normalize()
    return df


def add_spot_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    c = x["close"]
    h = x["high"]
    l = x["low"]
    o = x["open"]

    for n in (1, 2, 3, 5, 10, 15, 30):
        x[f"return_{n}m"] = c.pct_change(n)

    for span in (5, 9, 20, 50, 100, 200):
        ema = c.ewm(span=span, adjust=False).mean()
        x[f"ema_{span}"] = ema
        x[f"ema_{span}_dist"] = safe_div(c - ema, ema)
        x[f"ema_{span}_slope_3"] = ema.diff(3)

    prev = c.shift(1)
    tr = pd.concat(
        [h - l, (h - prev).abs(), (l - prev).abs()],
        axis=1,
    ).max(axis=1)

    x["true_range"] = tr
    x["atr_14"] = tr.rolling(14, min_periods=14).mean()
    x["atr_pct"] = safe_div(x["atr_14"], c)

    atr5 = tr.rolling(5, min_periods=5).mean()
    atr20 = tr.rolling(20, min_periods=20).mean()
    x["atr_ratio_5_20"] = safe_div(atr5, atr20)
    x["atr_expansion_slope"] = x["atr_ratio_5_20"].diff()

    rng = (h - l).replace(0, np.nan)
    x["body_to_range"] = safe_div((c - o).abs(), rng)
    x["close_location_in_range"] = safe_div(c - l, rng)

    r1 = c.pct_change()
    for w in (5, 10, 20, 30, 60):
        x[f"volatility_{w}"] = r1.rolling(w, min_periods=w).std()

    x["vol_expansion_5_20"] = safe_div(
        x["volatility_5"], x["volatility_20"]
    )

    x["velocity_1m"] = x["return_1m"]
    x["velocity_3m"] = safe_div(c.diff(3), x["atr_14"])
    x["velocity_5m"] = safe_div(c.diff(5), x["atr_14"])
    x["acceleration_1m"] = x["velocity_1m"].diff()
    x["acceleration_3m"] = x["velocity_3m"].diff()
    x["jerk_1m"] = x["acceleration_1m"].diff()

    for w in (5, 10, 15):
        path = c.diff().abs().rolling(w - 1, min_periods=w - 1).sum()
        displacement = c.diff(w - 1).abs()
        x[f"path_efficiency_{w}"] = safe_div(
            displacement, path
        ).clip(0, 1)

    sign = np.sign(c.diff()).fillna(0)
    grp = sign.ne(sign.shift()).cumsum()
    x["same_direction_sign"] = sign
    x["same_direction_run_length"] = np.where(
        sign == 0,
        0,
        sign.groupby(grp).cumcount() + 1,
    )

    g = x.groupby("date", sort=False)
    x["day_open"] = g["open"].transform("first")
    x["day_high"] = g["high"].cummax()
    x["day_low"] = g["low"].cummin()
    x["return_from_open"] = safe_div(c - x["day_open"], x["day_open"])
    x["distance_day_high_atr"] = safe_div(x["day_high"] - c, x["atr_14"])
    x["distance_day_low_atr"] = safe_div(c - x["day_low"], x["atr_14"])
    x["extension_ema20_atr"] = safe_div(c - x["ema_20"], x["atr_14"])

    add_available_flag(
        x,
        ["return_1m", "atr_14", "ema_20", "path_efficiency_5"],
        "spot_core_available",
    )

    return x.replace([np.inf, -np.inf], np.nan)


def completed_htf(
    spot: pd.DataFrame,
    minutes: int,
    prefix: str,
) -> pd.DataFrame:
    src = (
        spot[["timestamp", "open", "high", "low", "close"]]
        .dropna()
        .set_index("timestamp")
    )

    origin = (
        pd.Timestamp(src.index.min()).normalize()
        + pd.Timedelta(hours=9, minutes=15)
    )

    bars = src.resample(
        f"{minutes}min",
        origin=origin,
        label="right",
        closed="left",
    ).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        count=("close", "count"),
    )

    # Keep only fully completed bars.
    bars = bars[bars["count"] >= minutes].copy()

    bars["ema_fast"] = bars["close"].ewm(span=2, adjust=False).mean()
    bars["ema_slow"] = bars["close"].ewm(span=4, adjust=False).mean()
    bars["ret1"] = bars["close"].pct_change()
    bars["ema_spread"] = safe_div(
        bars["ema_fast"] - bars["ema_slow"],
        bars["ema_slow"],
    )

    bull = (
        (bars["ema_fast"] > bars["ema_slow"])
        & (bars["ret1"] > 0)
    )
    bear = (
        (bars["ema_fast"] < bars["ema_slow"])
        & (bars["ret1"] < 0)
    )

    bars["direction"] = np.select(
        [bull, bear],
        [1.0, -1.0],
        default=0.0,
    )

    changed = bars["direction"].ne(bars["direction"].shift())
    bars["bars_since_direction_change"] = (
        bars.groupby(changed.cumsum()).cumcount()
    )

    bars = bars.reset_index().rename(
        columns={"timestamp": "htf_close_time"}
    )

    decisions = spot[["timestamp"]].copy()

    # At decision at the next minute, only already completed HTF bars are visible.
    decisions["decision_time"] = (
        decisions["timestamp"] + pd.Timedelta(minutes=1)
    )

    mapped = pd.merge_asof(
        decisions.sort_values("decision_time"),
        bars.sort_values("htf_close_time"),
        left_on="decision_time",
        right_on="htf_close_time",
        direction="backward",
        allow_exact_matches=True,
    )

    out = pd.DataFrame(index=spot.index)
    out[f"{prefix}_direction"] = mapped["direction"].to_numpy()
    out[f"{prefix}_return_1"] = mapped["ret1"].to_numpy()
    out[f"{prefix}_ema_spread"] = mapped["ema_spread"].to_numpy()
    out[f"{prefix}_bars_since_direction_change"] = (
        mapped["bars_since_direction_change"].to_numpy()
    )
    out[f"{prefix}_available"] = (
        mapped["htf_close_time"]
        .notna()
        .astype("int8")
        .to_numpy()
    )

    return out


# ---------------------------------------------------------------------------
# CUDA IV + Greeks
# ---------------------------------------------------------------------------

def cp_norm_pdf(x):
    return cp.exp(-0.5 * x * x) / SQRT_2PI


def cp_norm_cdf(x):
    return 0.5 * (1.0 + erf(x / SQRT_2))


def gpu_iv_greeks_batch(
    price_np: np.ndarray,
    spot_np: np.ndarray,
    strike_np: np.ndarray,
    t_np: np.ndarray,
    is_call_np: np.ndarray,
    rate: float,
    q: float,
    iterations: int = 18,
):
    price = cp.asarray(price_np, dtype=cp.float64)
    S = cp.asarray(spot_np, dtype=cp.float64)
    K = cp.asarray(strike_np, dtype=cp.float64)
    T = cp.asarray(t_np, dtype=cp.float64)
    is_call = cp.asarray(is_call_np, dtype=cp.bool_)

    valid = (
        cp.isfinite(price)
        & cp.isfinite(S)
        & cp.isfinite(K)
        & cp.isfinite(T)
        & (price > 0)
        & (S > 0)
        & (K > 0)
        & (T > 0)
    )

    intrinsic = cp.where(
        is_call,
        cp.maximum(S - K, 0.0),
        cp.maximum(K - S, 0.0),
    )
    valid &= price >= (intrinsic - 1e-6)

    sigma = cp.full(price.shape, 0.25, dtype=cp.float64)
    r = float(rate)
    q = float(q)

    for _ in range(iterations):
        sqrtT = cp.sqrt(cp.maximum(T, 1e-12))
        sig = cp.clip(sigma, 1e-4, 5.0)

        d1 = (
            cp.log(S / K)
            + (r - q + 0.5 * sig * sig) * T
        ) / (sig * sqrtT)

        d2 = d1 - sig * sqrtT

        dq = cp.exp(-q * T)
        dr = cp.exp(-r * T)

        call_p = (
            S * dq * cp_norm_cdf(d1)
            - K * dr * cp_norm_cdf(d2)
        )

        put_p = (
            K * dr * cp_norm_cdf(-d2)
            - S * dq * cp_norm_cdf(-d1)
        )

        model_p = cp.where(is_call, call_p, put_p)

        vega_raw = S * dq * cp_norm_pdf(d1) * sqrtT

        usable = (
            valid
            & cp.isfinite(vega_raw)
            & (vega_raw > 1e-8)
        )

        step = cp.where(
            usable,
            (model_p - price) / vega_raw,
            0.0,
        )

        sigma = cp.clip(
            sigma - step,
            1e-4,
            5.0,
        )

    sqrtT = cp.sqrt(cp.maximum(T, 1e-12))
    d1 = (
        cp.log(S / K)
        + (r - q + 0.5 * sigma * sigma) * T
    ) / (sigma * sqrtT)

    d2 = d1 - sigma * sqrtT

    dq = cp.exp(-q * T)
    dr = cp.exp(-r * T)

    call_p = (
        S * dq * cp_norm_cdf(d1)
        - K * dr * cp_norm_cdf(d2)
    )

    put_p = (
        K * dr * cp_norm_cdf(-d2)
        - S * dq * cp_norm_cdf(-d1)
    )

    fitted = cp.where(is_call, call_p, put_p)

    tolerance = cp.maximum(
        0.25,
        price * 0.01,
    )

    converged = (
        valid
        & cp.isfinite(sigma)
        & (cp.abs(fitted - price) <= tolerance)
    )

    pdf = cp_norm_pdf(d1)

    delta = cp.where(
        is_call,
        dq * cp_norm_cdf(d1),
        dq * (cp_norm_cdf(d1) - 1.0),
    )

    gamma = (
        dq * pdf
        / (S * sigma * sqrtT)
    )

    theta_call = (
        -(S * dq * pdf * sigma) / (2.0 * sqrtT)
        - rate * K * dr * cp_norm_cdf(d2)
        + q * S * dq * cp_norm_cdf(d1)
    )

    theta_put = (
        -(S * dq * pdf * sigma) / (2.0 * sqrtT)
        + rate * K * dr * cp_norm_cdf(-d2)
        - q * S * dq * cp_norm_cdf(-d1)
    )

    theta = (
        cp.where(
            is_call,
            theta_call,
            theta_put,
        )
        / 365.0
    )

    vega = (
        S
        * dq
        * pdf
        * sqrtT
        / 100.0
    )

    nan = cp.nan

    sigma = cp.where(converged, sigma, nan)
    delta = cp.where(converged, delta, nan)
    gamma = cp.where(converged, gamma, nan)
    theta = cp.where(converged, theta, nan)
    vega = cp.where(converged, vega, nan)

    result = tuple(
        cp.asnumpy(v)
        for v in (
            sigma,
            delta,
            gamma,
            theta,
            vega,
        )
    )

    del price, S, K, T, is_call
    release_memory()

    return result


def add_iv_greeks_cuda(
    df: pd.DataFrame,
    rate: float,
    q: float,
    batch_rows: int,
) -> pd.DataFrame:
    x = df.copy()

    exp_close = (
        pd.to_datetime(
            x["expiry_date"],
            errors="coerce",
        ).dt.normalize()
        + pd.Timedelta(hours=15, minutes=30)
    )

    x["time_to_expiry_years"] = (
        (exp_close - x["timestamp"]).dt.total_seconds()
        / (365.0 * 24 * 3600)
    )

    x.loc[
        x["time_to_expiry_years"] <= 0,
        "time_to_expiry_years",
    ] = np.nan

    x["dte"] = (
        pd.to_datetime(x["expiry_date"]).dt.normalize()
        - x["timestamp"].dt.normalize()
    ).dt.total_seconds() / 86400.0

    x["moneyness"] = safe_div(
        x["spot_close"] - x["strike"],
        x["spot_close"],
    )

    x["abs_moneyness"] = (
        x["moneyness"].abs()
    )

    n = len(x)

    iv = np.full(n, np.nan)
    delta = np.full(n, np.nan)
    gamma = np.full(n, np.nan)
    theta = np.full(n, np.nan)
    vega = np.full(n, np.nan)

    for start in range(0, n, batch_rows):
        end = min(
            start + batch_rows,
            n,
        )

        sl = slice(start, end)

        vals = gpu_iv_greeks_batch(
            x["close"].iloc[sl].to_numpy(float),
            x["spot_close"].iloc[sl].to_numpy(float),
            x["strike"].iloc[sl].to_numpy(float),
            x["time_to_expiry_years"].iloc[sl].to_numpy(float),
            (
                x["option_type"]
                .iloc[sl]
                .to_numpy()
                == "CE"
            ),
            rate,
            q,
        )

        (
            iv[sl],
            delta[sl],
            gamma[sl],
            theta[sl],
            vega[sl],
        ) = vals

    x["implied_volatility"] = iv
    x["delta"] = delta
    x["gamma"] = gamma
    x["theta_per_day"] = theta
    x["vega_per_pct"] = vega

    g = x.groupby(
        "groww_symbol",
        sort=False,
    )

    x["iv_change_1m"] = (
        g["implied_volatility"].diff()
    )

    x["iv_change_3m"] = (
        g["implied_volatility"].diff(3)
    )

    x["delta_change_1m"] = (
        g["delta"].diff()
    )

    x["theta_burden"] = safe_div(
        x["theta_per_day"].abs(),
        x["close"],
    )

    x["gamma_responsiveness"] = (
        x["gamma"] * x["spot_close"]
    )

    add_available_flag(
        x,
        [
            "implied_volatility",
            "delta",
            "gamma",
            "theta_per_day",
            "vega_per_pct",
        ],
        "greeks_available",
    )

    return x.replace(
        [np.inf, -np.inf],
        np.nan,
    )


# ---------------------------------------------------------------------------
# Option-contract batch processing
# ---------------------------------------------------------------------------

def load_option_batch(
    files: Sequence[Path],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "groww_symbol",
        "strike",
        "option_type",
        "expiry_date",
    }

    for path in files:
        df = pd.read_parquet(path)

        if not required.issubset(df.columns):
            print(
                f"WARNING: skipping {path.name}; "
                f"missing columns="
                f"{sorted(required - set(df.columns))}"
            )
            continue

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        df["expiry_date"] = pd.to_datetime(
            df["expiry_date"],
            errors="coerce",
        )

        df = clean_market_values(df)

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    x = pd.concat(
        frames,
        ignore_index=True,
    )

    del frames
    gc.collect()

    x["option_type"] = (
        x["option_type"]
        .astype(str)
        .str.upper()
    )

    x = x[
        x["option_type"].isin(["CE", "PE"])
    ]

    x = (
        x.sort_values(
            ["groww_symbol", "timestamp"]
        )
        .drop_duplicates(
            ["groww_symbol", "timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return x


def add_option_dynamics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    x = df.copy()

    g = x.groupby(
        "groww_symbol",
        sort=False,
    )

    x["option_return_1m"] = (
        g["close"].pct_change()
    )

    x["option_return_3m"] = (
        g["close"].pct_change(3)
    )

    x["option_return_5m"] = (
        g["close"].pct_change(5)
    )

    x["premium_change_1m"] = (
        g["close"].diff()
    )

    x["premium_acceleration"] = (
        g["close"]
        .diff()
        .groupby(x["groww_symbol"])
        .diff()
    )

    x["volume_change_1m"] = (
        g["volume"].diff()
    )

    x["oi_change_1m"] = (
        g["open_interest"].diff()
    )

    x["oi_pct_change_1m"] = (
        g["open_interest"].pct_change()
    )

    vol3 = g["volume"].transform(
        lambda s:
        s.rolling(
            3,
            min_periods=3,
        ).mean()
    )

    vol10 = g["volume"].transform(
        lambda s:
        s.rolling(
            10,
            min_periods=10,
        ).mean()
    )

    x["volume_accel_3_10"] = (
        safe_div(vol3, vol10)
    )

    x["volume_accel_1_10"] = (
        safe_div(
            x["volume"],
            vol10,
        )
    )

    x["price_oi_product"] = (
        x["close"]
        * x["open_interest"]
    )

    x["price_oi_change"] = (
        x["premium_change_1m"]
        * x["oi_change_1m"]
    )

    x["price_return_x_oi_pct_change"] = (
        x["option_return_1m"]
        * x["oi_pct_change_1m"]
    )

    p = x["premium_change_1m"]
    oi = x["oi_change_1m"]

    x["long_buildup"] = (
        (p > 0) & (oi > 0)
    ).astype("int8")

    x["short_covering"] = (
        (p > 0) & (oi < 0)
    ).astype("int8")

    x["short_buildup"] = (
        (p < 0) & (oi > 0)
    ).astype("int8")

    x["long_unwinding"] = (
        (p < 0) & (oi < 0)
    ).astype("int8")

    add_available_flag(
        x,
        [
            "close",
            "volume",
            "open_interest",
        ],
        "option_market_available",
    )

    return x.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def merge_spot_context(
    options: pd.DataFrame,
    spot: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "timestamp",
        "close",
        "atr_14",
        "atr_ratio_5_20",
        "return_1m",
        "return_3m",
        "return_5m",
        "velocity_3m",
        "velocity_5m",
        "acceleration_1m",
        "path_efficiency_5",
        "extension_ema20_atr",
        "same_direction_sign",
        "same_direction_run_length",
        "htf_5m_direction",
        "htf_15m_direction",
        "htf_5m_return_1",
        "htf_15m_return_1",
        "htf_5m_bars_since_direction_change",
        "htf_15m_bars_since_direction_change",
    ]

    cols = [
        c for c in cols
        if c in spot.columns
    ]

    ctx = (
        spot[cols]
        .rename(
            columns={
                "close": "spot_close"
            }
        )
    )

    return options.merge(
        ctx,
        on="timestamp",
        how="left",
        validate="many_to_one",
    )


# ---------------------------------------------------------------------------
# DuckDB chain aggregation
# ---------------------------------------------------------------------------

def duckdb_connection(
    threads: int,
    memory_limit_gb: float,
    temp_dir: Path,
):
    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = duckdb.connect()

    con.execute(
        f"PRAGMA threads={int(threads)}"
    )

    con.execute(
        "PRAGMA preserve_insertion_order=false"
    )

    con.execute(
        f"PRAGMA memory_limit='{float(memory_limit_gb):.1f}GB'"
    )

    # DuckDB may spill to disk instead of blowing RAM.
    escaped = str(temp_dir).replace("'", "''")
    con.execute(
        f"PRAGMA temp_directory='{escaped}'"
    )

    return con


def build_chain_features(
    parts_glob: str,
    output_path: Path,
    chain_wings: int,
    strike_step: float,
    duckdb_threads: int,
    duckdb_memory_gb: float,
    temp_dir: Path,
) -> None:
    con = duckdb_connection(
        duckdb_threads,
        duckdb_memory_gb,
        temp_dir,
    )

    src = parts_glob.replace("'", "''")
    dst = str(output_path).replace("'", "''")

    sql = f"""
    COPY (
        WITH base AS (
            SELECT
                *,
                ROUND(
                    spot_close / {strike_step}
                ) * {strike_step}
                AS atm_strike,

                ROUND(
                    (
                        strike
                        - ROUND(
                            spot_close / {strike_step}
                        ) * {strike_step}
                    )
                    / {strike_step}
                ) AS strike_offset

            FROM read_parquet(
                '{src}'
            )
        ),

        w AS (
            SELECT *
            FROM base
            WHERE
                ABS(strike_offset)
                <= {int(chain_wings)}
        ),

        agg AS (
            SELECT
                timestamp,
                expiry_date,

                SUM(
                    CASE
                    WHEN option_type='CE'
                    THEN open_interest
                    ELSE NULL
                    END
                ) AS ce_oi_total,

                SUM(
                    CASE
                    WHEN option_type='PE'
                    THEN open_interest
                    ELSE NULL
                    END
                ) AS pe_oi_total,

                SUM(
                    CASE
                    WHEN option_type='CE'
                    THEN volume
                    ELSE NULL
                    END
                ) AS ce_volume_total,

                SUM(
                    CASE
                    WHEN option_type='PE'
                    THEN volume
                    ELSE NULL
                    END
                ) AS pe_volume_total,

                CASE
                WHEN SUM(
                    CASE
                    WHEN option_type='CE'
                    THEN open_interest
                    ELSE NULL
                    END
                ) > 0
                THEN
                    SUM(
                        CASE
                        WHEN option_type='CE'
                        THEN strike * open_interest
                        ELSE NULL
                        END
                    )
                    /
                    SUM(
                        CASE
                        WHEN option_type='CE'
                        THEN open_interest
                        ELSE NULL
                        END
                    )
                ELSE NULL
                END
                AS ce_oi_center,

                CASE
                WHEN SUM(
                    CASE
                    WHEN option_type='PE'
                    THEN open_interest
                    ELSE NULL
                    END
                ) > 0
                THEN
                    SUM(
                        CASE
                        WHEN option_type='PE'
                        THEN strike * open_interest
                        ELSE NULL
                        END
                    )
                    /
                    SUM(
                        CASE
                        WHEN option_type='PE'
                        THEN open_interest
                        ELSE NULL
                        END
                    )
                ELSE NULL
                END
                AS pe_oi_center,

                AVG(
                    CASE
                    WHEN option_type='CE'
                    THEN price_return_x_oi_pct_change
                    ELSE NULL
                    END
                )
                AS ce_price_oi_pressure,

                AVG(
                    CASE
                    WHEN option_type='PE'
                    THEN price_return_x_oi_pct_change
                    ELSE NULL
                    END
                )
                AS pe_price_oi_pressure

            FROM w
            GROUP BY
                timestamp,
                expiry_date
        ),

        lagged AS (
            SELECT
                *,

                ce_oi_center
                - LAG(ce_oi_center)
                  OVER(
                    PARTITION BY expiry_date
                    ORDER BY timestamp
                  )
                AS call_oi_center_change,

                pe_oi_center
                - LAG(pe_oi_center)
                  OVER(
                    PARTITION BY expiry_date
                    ORDER BY timestamp
                  )
                AS put_oi_center_change

            FROM agg
        )

        SELECT
            *,

            pe_oi_total
            / NULLIF(
                ce_oi_total,
                0
            )
            AS pcr_oi,

            pe_volume_total
            / NULLIF(
                ce_volume_total,
                0
            )
            AS pcr_volume,

            (
                COALESCE(
                    call_oi_center_change,
                    0
                )
                +
                COALESCE(
                    put_oi_center_change,
                    0
                )
            )
            / {strike_step}
            AS oi_migration_score,

            COALESCE(
                pe_price_oi_pressure,
                0
            )
            -
            COALESCE(
                ce_price_oi_pressure,
                0
            )
            AS chain_price_oi_bias,

            CASE
            WHEN
                ce_oi_total IS NOT NULL
                AND pe_oi_total IS NOT NULL
                AND ce_oi_center IS NOT NULL
                AND pe_oi_center IS NOT NULL
            THEN 1
            ELSE 0
            END
            AS chain_oi_available

        FROM lagged
    )
    TO '{dst}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    );
    """

    con.execute(sql)
    con.close()


def merge_chain_features(
    parts_glob: str,
    chain_path: Path,
    output_path: Path,
    duckdb_threads: int,
    duckdb_memory_gb: float,
    temp_dir: Path,
) -> None:
    con = duckdb_connection(
        duckdb_threads,
        duckdb_memory_gb,
        temp_dir,
    )

    src = parts_glob.replace("'", "''")
    chain = str(chain_path).replace("'", "''")
    dst = str(output_path).replace("'", "''")

    con.execute(
        f"""
        COPY (
            SELECT
                p.*,
                c.ce_oi_total,
                c.pe_oi_total,
                c.ce_volume_total,
                c.pe_volume_total,
                c.ce_oi_center,
                c.pe_oi_center,
                c.ce_price_oi_pressure,
                c.pe_price_oi_pressure,
                c.call_oi_center_change,
                c.put_oi_center_change,
                c.pcr_oi,
                c.pcr_volume,
                c.oi_migration_score,
                c.chain_price_oi_bias,
                c.chain_oi_available
            FROM
                read_parquet('{src}') p
            LEFT JOIN
                read_parquet('{chain}') c
            USING(
                timestamp,
                expiry_date
            )
        )
        TO '{dst}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
        """
    )

    con.close()


# ---------------------------------------------------------------------------
# Momentum / exhaustion
# ---------------------------------------------------------------------------

def _component(
    cond: pd.Series,
    available: pd.Series,
    index,
) -> pd.Series:
    s = pd.Series(
        np.nan,
        index=index,
        dtype=float,
    )
    mask = available.fillna(False)
    s.loc[mask] = (
        cond.loc[mask]
        .astype(float)
    )
    return s


def add_momentum_scores(
    df: pd.DataFrame,
) -> pd.DataFrame:
    x = df.copy()

    x["side_sign"] = np.where(
        x["option_type"] == "CE",
        1.0,
        -1.0,
    )

    side_cols = [
        "return_1m",
        "return_3m",
        "return_5m",
        "velocity_3m",
        "velocity_5m",
        "acceleration_1m",
        "htf_5m_direction",
        "htf_15m_direction",
        "htf_5m_return_1",
        "htf_15m_return_1",
        "extension_ema20_atr",
    ]

    for c in side_cols:
        if c in x.columns:
            x[c + "_side"] = (
                x[c] * x["side_sign"]
            )

    maturity = pd.DataFrame(
        index=x.index
    )

    maturity["htf5"] = _component(
        x["htf_5m_direction_side"] > 0,
        x["htf_5m_direction_side"].notna(),
        x.index,
    )

    maturity["htf15"] = _component(
        x["htf_15m_direction_side"] >= 0,
        x["htf_15m_direction_side"].notna(),
        x.index,
    )

    maturity["velocity"] = _component(
        x["velocity_3m_side"] > 0,
        x["velocity_3m_side"].notna(),
        x.index,
    )

    maturity["acceleration"] = _component(
        x["acceleration_1m_side"] > 0,
        x["acceleration_1m_side"].notna(),
        x.index,
    )

    maturity["path"] = _component(
        x["path_efficiency_5"] > 0.55,
        x["path_efficiency_5"].notna(),
        x.index,
    )

    maturity["volume"] = _component(
        x["volume_accel_1_10"] > 1.2,
        x["volume_accel_1_10"].notna(),
        x.index,
    )

    maturity["premium"] = _component(
        x["premium_acceleration"] > 0,
        x["premium_acceleration"].notna(),
        x.index,
    )

    maturity["extension"] = _component(
        x["extension_ema20_atr_side"] < 1.5,
        x["extension_ema20_atr_side"].notna(),
        x.index,
    )

    m_avail = maturity.notna().sum(axis=1)

    x["momentum_maturity_components_available"] = (
        m_avail.astype("int16")
    )

    x["momentum_maturity_score"] = (
        100.0
        * maturity.sum(
            axis=1,
            min_count=1,
        )
        / m_avail.replace(
            0,
            np.nan,
        )
    )

    x.loc[
        m_avail < 5,
        "momentum_maturity_score",
    ] = np.nan

    exhaustion = pd.DataFrame(
        index=x.index
    )

    exhaustion["overextended"] = _component(
        x["extension_ema20_atr_side"] > 1.5,
        x["extension_ema20_atr_side"].notna(),
        x.index,
    )

    exhaustion["long_run"] = _component(
        (
            x["same_direction_run_length"] >= 5
        )
        & (
            x["same_direction_sign"]
            == x["side_sign"]
        ),
        (
            x["same_direction_run_length"].notna()
            & x["same_direction_sign"].notna()
        ),
        x.index,
    )

    exhaustion["accel_fading"] = _component(
        x["acceleration_1m_side"] < 0,
        x["acceleration_1m_side"].notna(),
        x.index,
    )

    exhaustion["atr_extreme"] = _component(
        x["atr_ratio_5_20"] > 1.5,
        x["atr_ratio_5_20"].notna(),
        x.index,
    )

    exhaustion["premium_divergence"] = _component(
        (
            x["return_3m_side"] > 0
        )
        & (
            x["option_return_3m"] <= 0
        ),
        (
            x["return_3m_side"].notna()
            & x["option_return_3m"].notna()
        ),
        x.index,
    )

    exhaustion["iv_spike"] = _component(
        x["iv_change_3m"] > 0.10,
        x["iv_change_3m"].notna(),
        x.index,
    )

    e_avail = exhaustion.notna().sum(
        axis=1
    )

    x["exhaustion_components_available"] = (
        e_avail.astype("int16")
    )

    x["exhaustion_score"] = (
        100.0
        * exhaustion.sum(
            axis=1,
            min_count=1,
        )
        / e_avail.replace(
            0,
            np.nan,
        )
    )

    x.loc[
        e_avail < 4,
        "exhaustion_score",
    ] = np.nan

    add_available_flag(
        x,
        [
            "spot_close",
            "close",
            "strike",
            "expiry_date",
        ],
        "critical_market_available",
    )

    add_available_flag(
        x,
        [
            "volume",
            "open_interest",
        ],
        "liquidity_available",
    )

    add_available_flag(
        x,
        [
            "htf_5m_direction",
            "htf_15m_direction",
        ],
        "htf_available",
    )

    return x.replace(
        [np.inf, -np.inf],
        np.nan,
    )


# ---------------------------------------------------------------------------
# Final streaming pass
# ---------------------------------------------------------------------------

def process_final_stream(
    merged_path: Path,
    output_dir: Path,
    batch_rows: int,
) -> Dict[str, Any]:
    final_parts_dir = (
        output_dir / "final_parts"
    )

    final_parts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_file = pq.ParquetFile(
        merged_path
    )

    total_before = 0
    total_after = 0
    greeks_available = 0
    chain_available = 0
    inf_count = 0
    missing_counts: Dict[str, int] = {}

    for i, rb in enumerate(
        parquet_file.iter_batches(
            batch_size=batch_rows
        )
    ):
        df = rb.to_pandas()
        total_before += len(df)

        df = add_momentum_scores(df)

        keep = (
            df["timestamp"].notna()
            & df["groww_symbol"].notna()
            & df["spot_close"].notna()
            & (df["spot_close"] > 0)
            & df["close"].notna()
            & (df["close"] > 0)
            & df["strike"].notna()
            & (df["strike"] > 0)
            & df["expiry_date"].notna()
            & df["option_type"].isin(
                ["CE", "PE"]
            )
        )

        df = (
            df.loc[keep]
            .reset_index(drop=True)
        )

        total_after += len(df)

        if "greeks_available" in df.columns:
            greeks_available += int(
                df["greeks_available"]
                .fillna(0)
                .sum()
            )

        if "chain_oi_available" in df.columns:
            chain_available += int(
                df["chain_oi_available"]
                .fillna(0)
                .sum()
            )

        numeric = df.select_dtypes(
            include=[np.number]
        )

        inf_count += int(
            np.isinf(
                numeric.to_numpy()
            ).sum()
        )

        for col, n in df.isna().sum().items():
            missing_counts[col] = (
                missing_counts.get(
                    col,
                    0,
                )
                + int(n)
            )

        atomic_parquet(
            df,
            final_parts_dir
            / f"part_{i:05d}.parquet",
        )

        print(
            f"Final pass part {i:05d}: "
            f"{len(df):,} rows"
        )

        del df, rb
        release_memory()

    return {
        "rows_before": int(total_before),
        "rows_after": int(total_after),
        "rows_dropped": int(
            total_before - total_after
        ),
        "greeks_available_fraction": (
            greeks_available / total_after
            if total_after
            else None
        ),
        "chain_oi_available_fraction": (
            chain_available / total_after
            if total_after
            else None
        ),
        "numeric_inf_count": int(
            inf_count
        ),
        "top_missing_columns": dict(
            sorted(
                missing_counts.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:50]
        ),
    }


def combine_final_parts(
    final_parts_glob: str,
    output_path: Path,
    duckdb_threads: int,
    duckdb_memory_gb: float,
    temp_dir: Path,
) -> None:
    con = duckdb_connection(
        duckdb_threads,
        duckdb_memory_gb,
        temp_dir,
    )

    src = final_parts_glob.replace(
        "'",
        "''",
    )

    dst = str(output_path).replace(
        "'",
        "''",
    )

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM read_parquet(
                '{src}'
            )
        )
        TO '{dst}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
        """
    )

    con.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input-dir",
        default="data/banknifty_groww",
    )

    ap.add_argument(
        "--output-dir",
        default="data/banknifty_processed_stream",
    )

    ap.add_argument(
        "--files-per-batch",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--gpu-batch-rows",
        type=int,
        default=100000,
    )

    ap.add_argument(
        "--final-batch-rows",
        type=int,
        default=150000,
    )

    ap.add_argument(
        "--risk-free-rate",
        type=float,
        default=DEFAULT_RATE,
    )

    ap.add_argument(
        "--dividend-yield",
        type=float,
        default=DEFAULT_Q,
    )

    ap.add_argument(
        "--chain-wings",
        type=int,
        default=DEFAULT_CHAIN_WINGS,
    )

    ap.add_argument(
        "--strike-step",
        type=float,
        default=DEFAULT_STRIKE_STEP,
    )

    ap.add_argument(
        "--duckdb-threads",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--duckdb-memory-gb",
        type=float,
        default=4.0,
        help=(
            "Maximum DuckDB RAM before it spills to disk. "
            "Set below available system RAM."
        ),
    )

    ap.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Delete intermediate/output files "
            "and restart from scratch."
        ),
    )

    args = ap.parse_args()

    if args.files_per_batch < 1:
        raise SystemExit(
            "--files-per-batch must be >= 1"
        )

    if args.gpu_batch_rows < 1000:
        raise SystemExit(
            "--gpu-batch-rows is too small"
        )

    inp = Path(args.input_dir)
    out = Path(args.output_dir)

    spot_path = (
        inp
        / "underlying"
        / "banknifty_1min.parquet"
    )

    option_dir = (
        inp
        / "options"
        / "contracts"
    )

    if not spot_path.exists():
        raise SystemExit(
            f"Missing underlying file: "
            f"{spot_path}"
        )

    if not option_dir.exists():
        raise SystemExit(
            f"Missing option directory: "
            f"{option_dir}"
        )

    stage1_dir = (
        out / "stage1_option_parts"
    )

    final_parts_dir = (
        out / "final_parts"
    )

    temp_dir = (
        out / "duckdb_temp"
    )

    if args.fresh and out.exists():
        shutil.rmtree(out)

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage1_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = cuda_device_info()

    print(
        "CUDA device:",
        device["name"],
        f"| free={device['free_gb']:.2f} GB",
        f"| total={device['total_gb']:.2f} GB",
    )

    atomic_json(
        device,
        out / "cuda_info.json",
    )

    # ------------------------------------------------------------
    # STEP 1: UNDERLYING
    # ------------------------------------------------------------
    spot_features_path = (
        out
        / "banknifty_underlying_features.parquet"
    )

    if spot_features_path.exists():
        print(
            "1/7 Underlying features: cached"
        )
        spot = pd.read_parquet(
            spot_features_path
        )
    else:
        print(
            "1/7 Building underlying features..."
        )

        spot = load_spot(
            spot_path
        )

        spot = add_spot_features(
            spot
        )

        print(
            "2/7 Building causal 5m/15m features..."
        )

        spot = pd.concat(
            [
                spot,
                completed_htf(
                    spot,
                    5,
                    "htf_5m",
                ),
                completed_htf(
                    spot,
                    15,
                    "htf_15m",
                ),
            ],
            axis=1,
        )

        atomic_parquet(
            spot,
            spot_features_path,
        )

    # ------------------------------------------------------------
    # STEP 2: STREAM OPTION CONTRACTS
    # ------------------------------------------------------------
    files = sorted(
        option_dir.glob(
            "*.parquet"
        )
    )

    if not files:
        raise SystemExit(
            "No option parquet files found."
        )

    n_batches = math.ceil(
        len(files)
        / args.files_per_batch
    )

    print(
        f"3/7 Streaming {len(files):,} "
        f"option files in {n_batches:,} batches..."
    )

    for batch_idx, start in enumerate(
        range(
            0,
            len(files),
            args.files_per_batch,
        )
    ):
        target = (
            stage1_dir
            / f"part_{batch_idx:05d}.parquet"
        )

        if target.exists():
            print(
                f"  [{batch_idx+1}/{n_batches}] "
                "cached"
            )
            continue

        batch_files = files[
            start:
            start + args.files_per_batch
        ]

        print(
            f"  [{batch_idx+1}/{n_batches}] "
            f"loading {len(batch_files)} files..."
        )

        options = load_option_batch(
            batch_files
        )

        if options.empty:
            print(
                "    empty batch; skipped"
            )
            continue

        options = add_option_dynamics(
            options
        )

        options = merge_spot_context(
            options,
            spot,
        )

        options = add_iv_greeks_cuda(
            options,
            args.risk_free_rate,
            args.dividend_yield,
            args.gpu_batch_rows,
        )

        atomic_parquet(
            options,
            target,
        )

        print(
            f"    saved {len(options):,} rows"
        )

        del options
        release_memory()

    # Spot no longer needed after stage1.
    del spot
    release_memory()

    # ------------------------------------------------------------
    # STEP 3: CHAIN FEATURES
    # ------------------------------------------------------------
    stage1_glob = str(
        stage1_dir
        / "*.parquet"
    )

    chain_path = (
        out
        / "chain_features.parquet"
    )

    print(
        "4/7 Building chain OI/PCR/migration "
        "features with DuckDB..."
    )

    if not chain_path.exists():
        build_chain_features(
            stage1_glob,
            chain_path,
            args.chain_wings,
            args.strike_step,
            args.duckdb_threads,
            args.duckdb_memory_gb,
            temp_dir,
        )
    else:
        print(
            "    chain_features.parquet cached"
        )

    # ------------------------------------------------------------
    # STEP 4: JOIN CHAIN FEATURES
    # ------------------------------------------------------------
    merged_path = (
        out
        / "banknifty_option_features_full.parquet"
    )

    print(
        "5/7 Joining chain features..."
    )

    if not merged_path.exists():
        merge_chain_features(
            stage1_glob,
            chain_path,
            merged_path,
            args.duckdb_threads,
            args.duckdb_memory_gb,
            temp_dir,
        )
    else:
        print(
            "    full feature parquet cached"
        )

    # ------------------------------------------------------------
    # STEP 5: FINAL STREAMING SCORE PASS
    # ------------------------------------------------------------
    quality_path = (
        out / "quality_report.json"
    )

    print(
        "6/7 Streaming final momentum/"
        "exhaustion pass..."
    )

    if final_parts_dir.exists():
        # Resume: if final parts already exist,
        # restart final pass cleanly to avoid
        # accidental duplicates.
        shutil.rmtree(
            final_parts_dir
        )

    final_stats = process_final_stream(
        merged_path,
        out,
        args.final_batch_rows,
    )

    # ------------------------------------------------------------
    # STEP 6: COMBINE FINAL PARTS
    # ------------------------------------------------------------
    final_path = (
        out
        / "banknifty_option_features_model_ready.parquet"
    )

    print(
        "7/7 Combining final parts..."
    )

    combine_final_parts(
        str(
            final_parts_dir
            / "*.parquet"
        ),
        final_path,
        args.duckdb_threads,
        args.duckdb_memory_gb,
        temp_dir,
    )

    report = {
        "cuda": device,
        "underlying_rows": int(
            pq.ParquetFile(
                spot_features_path
            ).metadata.num_rows
        ),
        "option_rows_full": int(
            pq.ParquetFile(
                merged_path
            ).metadata.num_rows
        ),
        "option_rows_model_ready": int(
            pq.ParquetFile(
                final_path
            ).metadata.num_rows
        ),
        "greeks_available_fraction": (
            final_stats[
                "greeks_available_fraction"
            ]
        ),
        "chain_oi_available_fraction": (
            final_stats[
                "chain_oi_available_fraction"
            ]
        ),
        "filter_report": {
            "rows_before": (
                final_stats["rows_before"]
            ),
            "rows_after": (
                final_stats["rows_after"]
            ),
            "rows_dropped": (
                final_stats["rows_dropped"]
            ),
            "numeric_inf_count": (
                final_stats[
                    "numeric_inf_count"
                ]
            ),
            "top_missing_columns": (
                final_stats[
                    "top_missing_columns"
                ]
            ),
        },
        "settings": {
            "files_per_batch": (
                args.files_per_batch
            ),
            "gpu_batch_rows": (
                args.gpu_batch_rows
            ),
            "final_batch_rows": (
                args.final_batch_rows
            ),
            "duckdb_threads": (
                args.duckdb_threads
            ),
            "duckdb_memory_gb": (
                args.duckdb_memory_gb
            ),
            "risk_free_rate": (
                args.risk_free_rate
            ),
            "dividend_yield": (
                args.dividend_yield
            ),
            "chain_wings": (
                args.chain_wings
            ),
            "strike_step": (
                args.strike_step
            ),
        },
    }

    atomic_json(
        report,
        quality_path,
    )

    print()
    print("=" * 72)
    print("COMPLETED")
    print("=" * 72)

    print(
        "Full feature dataset:",
        merged_path,
    )

    print(
        "Model-ready dataset:",
        final_path,
    )

    print(
        "Quality report:",
        quality_path,
    )

    print()
    print(
        "Greeks available:",
        report[
            "greeks_available_fraction"
        ],
    )

    print(
        "Chain OI available:",
        report[
            "chain_oi_available_fraction"
        ],
    )

    print(
        "Rows dropped:",
        report[
            "filter_report"
        ][
            "rows_dropped"
        ],
    )

    print(
        "Numeric inf count:",
        report[
            "filter_report"
        ][
            "numeric_inf_count"
        ],
    )


if __name__ == "__main__":
    main()
