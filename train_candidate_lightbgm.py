#!/usr/bin/env python3
"""
CUDA/GPU LightGBM training for BANKNIFTY option candidates.

Default input
-------------
data/banknifty_candidates/candidate_dataset.parquet

Models
------
Separate CE and PE binary classifiers.

Default target
--------------
CE: label_ce_hit_40_before_stop
PE: label_pe_hit_40_before_stop

Chronological split
-------------------
Train      : <= 2025-12-31
Validation : 2026-01-01 .. 2026-04-30
Test       : 2026-05-01 .. 2026-08-31

Important leakage protection
----------------------------
- ALL columns beginning with "label_" are excluded from features.
- identifiers/symbols/expiry/timestamps are excluded from model features.
- future labels are used only as y.
- no random row split.

GPU behavior
------------
The script tries:
    device_type="cuda"
then:
    device_type="gpu"
then falls back to CPU only if the installed LightGBM lacks GPU/CUDA support.

Install
-------
pip install lightgbm pandas numpy pyarrow scikit-learn

For true CUDA LightGBM you need a LightGBM build compiled with CUDA support.
A normal pip wheel may support OpenCL "gpu" but not "cuda".

Run
---
python3 train_candidate_lightgbm.py \
    --input-file data/banknifty_candidates/candidate_dataset.parquet \
    --output-dir data/banknifty_lgbm \
    --target 40 \
    --side both
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import lightgbm as lgb
except Exception as exc:
    raise SystemExit(
        "LightGBM is required. Install with: pip install lightgbm\n"
        f"Import error: {exc}"
    )

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IDENTIFIER_EXACT = {
    "timestamp",
    "ce_groww_symbol",
    "pe_groww_symbol",
    "ce_expiry_date",
    "pe_expiry_date",
    "ce_option_type",
    "pe_option_type",
}

IDENTIFIER_SUFFIXES = (
    "_groww_symbol",
    "_expiry_date",
    "_option_type",
)

DEFAULT_TRAIN_END = "2025-12-31 23:59:59"
DEFAULT_VALID_START = "2026-01-01 00:00:00"
DEFAULT_VALID_END = "2026-04-30 23:59:59"
DEFAULT_TEST_START = "2026-05-01 00:00:00"
DEFAULT_TEST_END = "2026-08-31 23:59:59"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    m = np.isfinite(y_true) & np.isfinite(y_prob)
    if m.sum() < 2 or len(np.unique(y_true[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y_true[m], y_prob[m]))


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    m = np.isfinite(y_true) & np.isfinite(y_prob)
    if m.sum() < 2:
        return float("nan")
    return float(average_precision_score(y_true[m], y_prob[m]))


def metric_block(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    yt = y_true[mask].astype(int)
    yp = y_prob[mask]

    if len(yt) == 0:
        return {
            "rows": 0,
            "positive_rate": None,
            "roc_auc": None,
            "pr_auc": None,
        }

    pred = (yp >= threshold).astype(int)

    out = {
        "rows": int(len(yt)),
        "positive_rate": float(np.mean(yt)),
        "roc_auc": safe_auc(yt, yp),
        "pr_auc": safe_ap(yt, yp),
        "threshold": float(threshold),
        "precision": float(precision_score(yt, pred, zero_division=0)),
        "recall": float(recall_score(yt, pred, zero_division=0)),
        "f1": float(f1_score(yt, pred, zero_division=0)),
    }

    try:
        tn, fp, fn, tp = confusion_matrix(
            yt, pred, labels=[0, 1]
        ).ravel()
        out["confusion_matrix"] = {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    except Exception:
        out["confusion_matrix"] = None

    return out


def top_fraction_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fractions: Sequence[float] = (0.05, 0.10, 0.20),
) -> Dict[str, Any]:
    df = pd.DataFrame({
        "y": np.asarray(y_true, dtype=float),
        "p": np.asarray(y_prob, dtype=float),
    }).dropna()

    if df.empty:
        return {}

    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    baseline = float(df["y"].mean())

    result: Dict[str, Any] = {
        "baseline_positive_rate": baseline,
    }

    for frac in fractions:
        n = max(1, int(math.ceil(len(df) * frac)))
        top = df.iloc[:n]
        rate = float(top["y"].mean())

        key = f"top_{int(frac*100)}pct"
        result[key] = {
            "rows": int(n),
            "success_rate": rate,
            "lift": (
                rate / baseline
                if baseline > 0
                else None
            ),
            "probability_min": float(top["p"].min()),
            "probability_mean": float(top["p"].mean()),
        }

    return result


def decile_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    d = pd.DataFrame({
        "y": np.asarray(y_true, dtype=float),
        "probability": np.asarray(y_prob, dtype=float),
    }).dropna()

    if d.empty:
        return pd.DataFrame()

    # Highest probability = decile 10.
    rank = d["probability"].rank(
        method="first",
        pct=True,
    )
    d["decile"] = np.ceil(rank * 10).clip(1, 10).astype(int)

    rows = []
    baseline = float(d["y"].mean())

    for dec in range(1, 11):
        sub = d[d["decile"] == dec]

        rows.append({
            "decile": dec,
            "rows": int(len(sub)),
            "mean_probability": (
                float(sub["probability"].mean())
                if len(sub)
                else np.nan
            ),
            "min_probability": (
                float(sub["probability"].min())
                if len(sub)
                else np.nan
            ),
            "max_probability": (
                float(sub["probability"].max())
                if len(sub)
                else np.nan
            ),
            "success_rate": (
                float(sub["y"].mean())
                if len(sub)
                else np.nan
            ),
            "lift_vs_baseline": (
                float(sub["y"].mean()) / baseline
                if len(sub) and baseline > 0
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


def probability_bin_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    d = pd.DataFrame({
        "y": np.asarray(y_true, dtype=float),
        "probability": np.asarray(y_prob, dtype=float),
    }).dropna()

    if d.empty:
        return pd.DataFrame()

    bins = np.linspace(0.0, 1.0, 11)
    d["probability_bin"] = pd.cut(
        d["probability"],
        bins=bins,
        include_lowest=True,
    )

    rows = []
    for label, sub in d.groupby(
        "probability_bin",
        observed=True,
        sort=True,
    ):
        rows.append({
            "probability_bin": str(label),
            "rows": int(len(sub)),
            "mean_predicted_probability": float(
                sub["probability"].mean()
            ),
            "observed_success_rate": float(
                sub["y"].mean()
            ),
        })

    return pd.DataFrame(rows)


def monthly_metrics(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    d = pd.DataFrame({
        "timestamp": pd.to_datetime(
            timestamps,
            errors="coerce",
        ),
        "y": y_true,
        "p": y_prob,
    }).dropna()

    if d.empty:
        return pd.DataFrame()

    d["month"] = d["timestamp"].dt.to_period("M").astype(str)

    rows = []
    for month, sub in d.groupby("month", sort=True):
        m = metric_block(
            sub["y"].to_numpy(),
            sub["p"].to_numpy(),
        )
        top = top_fraction_metrics(
            sub["y"].to_numpy(),
            sub["p"].to_numpy(),
            fractions=(0.10,),
        )

        rows.append({
            "month": month,
            "rows": m["rows"],
            "positive_rate": m["positive_rate"],
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"],
            "top10_success_rate": (
                top.get("top_10pct", {}).get(
                    "success_rate"
                )
            ),
            "top10_lift": (
                top.get("top_10pct", {}).get(
                    "lift"
                )
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_feature_columns(
    input_file: Path,
    side: str,
    target_col: str,
) -> Tuple[List[str], List[str]]:
    pf = pq.ParquetFile(input_file)
    all_cols = pf.schema.names

    excluded = []
    features = []

    opposite_prefix = "pe_" if side == "ce" else "ce_"

    for col in all_cols:
        if col == target_col:
            excluded.append(col)
            continue

        if col.startswith("label_"):
            excluded.append(col)
            continue

        if col in IDENTIFIER_EXACT:
            excluded.append(col)
            continue

        if col.endswith(IDENTIFIER_SUFFIXES):
            excluded.append(col)
            continue

        if col == "timestamp":
            excluded.append(col)
            continue

        # Keep common underlying/chain context and THIS side's option features.
        # Exclude opposite-side option contract-specific fields to avoid the
        # model becoming a diffuse two-contract state classifier.
        if col.startswith(opposite_prefix):
            # Preserve common chain columns that happen to start ce_/pe_.
            if col not in {
                "ce_oi_total",
                "pe_oi_total",
                "ce_oi_center",
                "pe_oi_center",
                "ce_volume_total",
                "pe_volume_total",
                "ce_price_oi_pressure",
                "pe_price_oi_pressure",
            }:
                excluded.append(col)
                continue

        features.append(col)

    return features, excluded


def load_columns_streaming(
    input_file: Path,
    columns: Sequence[str],
    batch_rows: int,
) -> pd.DataFrame:
    pf = pq.ParquetFile(input_file)

    frames = []
    for rb in pf.iter_batches(
        batch_size=batch_rows,
        columns=list(columns),
    ):
        frames.append(rb.to_pandas())

    if not frames:
        return pd.DataFrame()

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    del frames
    gc.collect()

    return out


def sanitize_features(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    kept = []
    dropped: Dict[str, str] = {}

    for col in feature_cols:
        if col not in df.columns:
            dropped[col] = "missing_from_loaded_data"
            continue

        s = df[col]

        # LightGBM needs numeric/bool/category. Convert booleans and numeric-like.
        if pd.api.types.is_bool_dtype(s):
            df[col] = s.astype("int8")

        elif not pd.api.types.is_numeric_dtype(s):
            converted = pd.to_numeric(
                s,
                errors="coerce",
            )

            # If essentially nonnumeric, exclude.
            if converted.notna().sum() == 0:
                dropped[col] = "nonnumeric"
                continue

            df[col] = converted

        finite_non_null = df[col].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        df[col] = finite_non_null

        if finite_non_null.notna().sum() == 0:
            dropped[col] = "all_null"
            continue

        nunique = finite_non_null.nunique(
            dropna=True
        )

        if nunique <= 1:
            dropped[col] = "constant"
            continue

        kept.append(col)

    return df, kept, dropped


# ---------------------------------------------------------------------------
# GPU / CUDA detection
# ---------------------------------------------------------------------------

def try_train_device(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    base_params: Dict[str, Any],
    preferred_device: str,
    gpu_platform_id: int,
    gpu_device_id: int,
):
    """
    Try CUDA first, then OpenCL GPU, then CPU.
    """
    attempts: List[Tuple[str, Dict[str, Any]]] = []

    if preferred_device in ("auto", "cuda"):
        attempts.append((
            "cuda",
            {
                "device_type": "cuda",
            },
        ))

    if preferred_device in ("auto", "gpu"):
        attempts.append((
            "gpu",
            {
                "device_type": "gpu",
                "gpu_platform_id": gpu_platform_id,
                "gpu_device_id": gpu_device_id,
            },
        ))

    if preferred_device in ("auto", "cpu"):
        attempts.append((
            "cpu",
            {
                "device_type": "cpu",
            },
        ))

    errors = {}

    for device_name, device_params in attempts:
        print(
            f"Trying LightGBM device: {device_name}",
            flush=True,
        )

        params = {
            **base_params,
            **device_params,
        }

        model = lgb.LGBMClassifier(
            **params
        )

        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[
                    (X_train, y_train),
                    (X_valid, y_valid),
                ],
                eval_names=[
                    "train",
                    "validation",
                ],
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=100,
                        verbose=True,
                    ),
                    lgb.log_evaluation(
                        period=50
                    ),
                ],
            )

            print(
                f"LightGBM training device selected: "
                f"{device_name.upper()}",
                flush=True,
            )
            return model, device_name, errors

        except Exception as exc:
            errors[device_name] = str(exc)
            print(
                f"{device_name.upper()} unavailable/failed: "
                f"{exc}",
                flush=True,
            )

    raise RuntimeError(
        "All requested LightGBM devices failed:\n"
        + json.dumps(errors, indent=2)
    )


# ---------------------------------------------------------------------------
# Train one side
# ---------------------------------------------------------------------------

def train_side(
    input_file: Path,
    output_dir: Path,
    side: str,
    target_points: int,
    batch_rows: int,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
    preferred_device: str,
    gpu_platform_id: int,
    gpu_device_id: int,
    num_threads: int,
    seed: int,
    class_weight_mode: str,
) -> Dict[str, Any]:

    side = side.lower()
    target_col = (
        f"label_{side}_hit_{int(target_points)}_before_stop"
    )

    print()
    print("=" * 72)
    print(
        f"TRAINING {side.upper()} MODEL | "
        f"TARGET={target_col}"
    )
    print("=" * 72)

    feature_cols, excluded = discover_feature_columns(
        input_file,
        side,
        target_col,
    )

    pf = pq.ParquetFile(input_file)
    all_cols = set(pf.schema.names)

    if target_col not in all_cols:
        raise ValueError(
            f"Target column missing: {target_col}"
        )

    load_cols = [
        "timestamp",
        target_col,
        *feature_cols,
    ]

    # remove duplicates preserving order
    seen = set()
    load_cols = [
        c for c in load_cols
        if not (c in seen or seen.add(c))
    ]

    print(
        f"Loading {len(load_cols):,} required columns...",
        flush=True,
    )

    df = load_columns_streaming(
        input_file,
        load_cols,
        batch_rows,
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df[target_col] = pd.to_numeric(
        df[target_col],
        errors="coerce",
    )

    # Only rows with valid target can train/evaluate.
    df = df[
        df["timestamp"].notna()
        & df[target_col].isin([0, 1])
    ].copy()

    df, feature_cols, dropped_features = sanitize_features(
        df,
        feature_cols,
    )

    print(
        f"Usable features: {len(feature_cols):,}"
    )

    if len(feature_cols) == 0:
        raise RuntimeError(
            "No usable numeric model features remain."
        )

    train_end_ts = pd.Timestamp(train_end)
    valid_start_ts = pd.Timestamp(valid_start)
    valid_end_ts = pd.Timestamp(valid_end)
    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    train_mask = (
        df["timestamp"] <= train_end_ts
    )
    valid_mask = (
        (df["timestamp"] >= valid_start_ts)
        & (df["timestamp"] <= valid_end_ts)
    )
    test_mask = (
        (df["timestamp"] >= test_start_ts)
        & (df["timestamp"] <= test_end_ts)
    )

    train = df.loc[train_mask].copy()
    valid = df.loc[valid_mask].copy()
    test = df.loc[test_mask].copy()

    if train.empty or valid.empty or test.empty:
        raise RuntimeError(
            f"Empty chronological split: "
            f"train={len(train)}, valid={len(valid)}, test={len(test)}"
        )

    X_train = train[feature_cols]
    y_train = train[target_col].astype(int).to_numpy()

    X_valid = valid[feature_cols]
    y_valid = valid[target_col].astype(int).to_numpy()

    X_test = test[feature_cols]
    y_test = test[target_col].astype(int).to_numpy()

    train_pos = float(np.mean(y_train))
    valid_pos = float(np.mean(y_valid))
    test_pos = float(np.mean(y_test))

    print(
        f"Rows: train={len(train):,}, "
        f"valid={len(valid):,}, "
        f"test={len(test):,}"
    )
    print(
        f"Positive rate: train={train_pos:.4f}, "
        f"valid={valid_pos:.4f}, "
        f"test={test_pos:.4f}"
    )

    scale_pos_weight = 1.0

    if class_weight_mode == "balanced":
        positives = float(np.sum(y_train == 1))
        negatives = float(np.sum(y_train == 0))
        if positives > 0:
            scale_pos_weight = negatives / positives

    base_params = {
        "objective": "binary",
        "n_estimators": 5000,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.2,
        "reg_lambda": 1.0,
        "max_bin": 255,
        "random_state": seed,
        "n_jobs": num_threads,
        "verbosity": -1,
        "scale_pos_weight": scale_pos_weight,
    }

    model, device_used, device_errors = try_train_device(
        X_train,
        y_train,
        X_valid,
        y_valid,
        base_params,
        preferred_device,
        gpu_platform_id,
        gpu_device_id,
    )

    p_train = model.predict_proba(
        X_train,
        num_iteration=model.best_iteration_,
    )[:, 1]

    p_valid = model.predict_proba(
        X_valid,
        num_iteration=model.best_iteration_,
    )[:, 1]

    p_test = model.predict_proba(
        X_test,
        num_iteration=model.best_iteration_,
    )[:, 1]

    metrics = {
        "side": side.upper(),
        "target": target_col,
        "device_used": device_used,
        "device_attempt_errors": device_errors,
        "best_iteration": (
            int(model.best_iteration_)
            if model.best_iteration_
            else None
        ),
        "feature_count": int(len(feature_cols)),
        "class_weight_mode": class_weight_mode,
        "scale_pos_weight": float(scale_pos_weight),
        "split": {
            "train_end": str(train_end_ts),
            "valid_start": str(valid_start_ts),
            "valid_end": str(valid_end_ts),
            "test_start": str(test_start_ts),
            "test_end": str(test_end_ts),
        },
        "train": metric_block(
            y_train,
            p_train,
        ),
        "validation": metric_block(
            y_valid,
            p_valid,
        ),
        "test": metric_block(
            y_test,
            p_test,
        ),
        "validation_ranking": top_fraction_metrics(
            y_valid,
            p_valid,
        ),
        "test_ranking": top_fraction_metrics(
            y_test,
            p_test,
        ),
        "dropped_features": dropped_features,
    }

    side_dir = output_dir / side
    side_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Model
    model.booster_.save_model(
        str(
            side_dir
            / f"{side}_hit{target_points}_lightgbm.txt"
        )
    )

    # Feature importance
    fi = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.booster_.feature_importance(
            importance_type="gain"
        ),
        "split": model.booster_.feature_importance(
            importance_type="split"
        ),
    })

    total_gain = fi["gain"].sum()

    fi["gain_pct"] = (
        fi["gain"] / total_gain * 100.0
        if total_gain > 0
        else 0.0
    )

    fi = fi.sort_values(
        "gain",
        ascending=False,
    )

    fi.to_csv(
        side_dir / "feature_importance.csv",
        index=False,
    )

    # Validation/test predictions
    valid_pred = pd.DataFrame({
        "timestamp": valid["timestamp"].to_numpy(),
        "target": y_valid,
        "probability": p_valid,
    })

    test_pred = pd.DataFrame({
        "timestamp": test["timestamp"].to_numpy(),
        "target": y_test,
        "probability": p_test,
    })

    valid_pred.to_parquet(
        side_dir / "validation_predictions.parquet",
        index=False,
    )

    test_pred.to_parquet(
        side_dir / "test_predictions.parquet",
        index=False,
    )

    valid_deciles = decile_table(
        y_valid,
        p_valid,
    )

    test_deciles = decile_table(
        y_test,
        p_test,
    )

    valid_deciles.to_csv(
        side_dir / "validation_deciles.csv",
        index=False,
    )

    test_deciles.to_csv(
        side_dir / "test_deciles.csv",
        index=False,
    )

    probability_bin_table(
        y_valid,
        p_valid,
    ).to_csv(
        side_dir / "validation_probability_bins.csv",
        index=False,
    )

    probability_bin_table(
        y_test,
        p_test,
    ).to_csv(
        side_dir / "test_probability_bins.csv",
        index=False,
    )

    monthly_metrics(
        test["timestamp"],
        y_test,
        p_test,
    ).to_csv(
        side_dir / "test_monthly_metrics.csv",
        index=False,
    )

    # Save feature list explicitly.
    atomic_json(
        {
            "side": side.upper(),
            "target": target_col,
            "features": feature_cols,
            "excluded_columns": excluded,
            "dropped_features": dropped_features,
        },
        side_dir / "feature_manifest.json",
    )

    atomic_json(
        metrics,
        side_dir / "metrics.json",
    )

    print()
    print(
        f"{side.upper()} TEST ROC-AUC: "
        f"{metrics['test']['roc_auc']:.4f}"
    )
    print(
        f"{side.upper()} TEST PR-AUC: "
        f"{metrics['test']['pr_auc']:.4f}"
    )

    for key in (
        "top_5pct",
        "top_10pct",
        "top_20pct",
    ):
        r = metrics["test_ranking"].get(
            key,
            {},
        )
        if r:
            print(
                f"{key}: success="
                f"{r['success_rate']:.4f}, "
                f"lift={r['lift']:.3f}x"
            )

    # Free big frames before next side.
    del (
        df,
        train,
        valid,
        test,
        X_train,
        X_valid,
        X_test,
        p_train,
        p_valid,
        p_test,
    )
    gc.collect()

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input-file",
        default=(
            "data/banknifty_candidates/"
            "candidate_dataset.parquet"
        ),
    )

    ap.add_argument(
        "--output-dir",
        default="data/banknifty_lgbm",
    )

    ap.add_argument(
        "--target",
        type=int,
        choices=[20, 30, 40],
        default=40,
        help=(
            "Premium target for binary label: "
            "+20/+30/+40 before -20."
        ),
    )

    ap.add_argument(
        "--side",
        choices=["ce", "pe", "both"],
        default="both",
    )

    ap.add_argument(
        "--batch-rows",
        type=int,
        default=150000,
    )

    ap.add_argument(
        "--train-end",
        default=DEFAULT_TRAIN_END,
    )

    ap.add_argument(
        "--valid-start",
        default=DEFAULT_VALID_START,
    )

    ap.add_argument(
        "--valid-end",
        default=DEFAULT_VALID_END,
    )

    ap.add_argument(
        "--test-start",
        default=DEFAULT_TEST_START,
    )

    ap.add_argument(
        "--test-end",
        default=DEFAULT_TEST_END,
    )

    ap.add_argument(
        "--device",
        choices=["auto", "cuda", "gpu", "cpu"],
        default="auto",
        help=(
            "auto tries CUDA -> OpenCL GPU -> CPU."
        ),
    )

    ap.add_argument(
        "--gpu-platform-id",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--gpu-device-id",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--num-threads",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help=(
            "For ranking experiments, 'none' is a good first run. "
            "'balanced' uses negative/positive scale_pos_weight."
        ),
    )

    ap.add_argument(
        "--fresh",
        action="store_true",
    )

    args = ap.parse_args()

    input_file = Path(
        args.input_file
    )

    output_dir = Path(
        args.output_dir
    )

    if not input_file.exists():
        raise SystemExit(
            f"Input file not found: {input_file}"
        )

    if args.fresh and output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"LightGBM version: {lgb.__version__}"
    )

    print(
        f"Input: {input_file}"
    )

    pf = pq.ParquetFile(
        input_file
    )

    print(
        f"Candidate rows: "
        f"{pf.metadata.num_rows:,}"
    )

    sides = (
        ["ce", "pe"]
        if args.side == "both"
        else [args.side]
    )

    summary = {
        "input_file": str(
            input_file
        ),
        "target_points": int(
            args.target
        ),
        "requested_device": args.device,
        "models": {},
    }

    for side in sides:
        metrics = train_side(
            input_file=input_file,
            output_dir=output_dir,
            side=side,
            target_points=args.target,
            batch_rows=args.batch_rows,
            train_end=args.train_end,
            valid_start=args.valid_start,
            valid_end=args.valid_end,
            test_start=args.test_start,
            test_end=args.test_end,
            preferred_device=args.device,
            gpu_platform_id=args.gpu_platform_id,
            gpu_device_id=args.gpu_device_id,
            num_threads=args.num_threads,
            seed=args.seed,
            class_weight_mode=args.class_weight,
        )

        summary["models"][
            side.upper()
        ] = metrics

    atomic_json(
        summary,
        output_dir
        / "training_summary.json",
    )

    print()
    print("=" * 72)
    print(
        "LIGHTGBM CANDIDATE TRAINING COMPLETE"
    )
    print("=" * 72)

    for side, m in summary[
        "models"
    ].items():
        print(
            f"{side}: device={m['device_used']}, "
            f"test_auc={m['test']['roc_auc']:.4f}, "
            f"test_pr_auc={m['test']['pr_auc']:.4f}, "
            f"top10_lift="
            f"{m['test_ranking'].get('top_10pct',{}).get('lift')}"
        )

    print(
        "Summary:",
        output_dir / "training_summary.json"
    )


if __name__ == "__main__":
    main()
