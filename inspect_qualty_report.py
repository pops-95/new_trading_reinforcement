#!/usr/bin/env python3
"""
Inspect BANKNIFTY preprocessing quality report.

Reads:
    data/banknifty_processed/quality_report.json

Optionally also reads:
    data/banknifty_processed/feature_manifest.json

Checks:
- greeks_available_fraction
- chain_oi_available_fraction
- rows dropped by critical filtering
- top missing columns
- numeric_inf_count
- feature missingness from feature_manifest.json

Outputs clear PASS / WARN / FAIL statuses and exits non-zero on FAIL.

Example:
    python3 inspect_quality_report.py

Custom paths:
    python3 inspect_quality_report.py \
        --quality-report data/banknifty_processed/quality_report.json \
        --feature-manifest data/banknifty_processed/feature_manifest.json

Suggested default thresholds:
- Greeks availability:
    PASS >= 0.80
    WARN 0.60-0.80
    FAIL < 0.60

- Chain OI availability:
    PASS >= 0.85
    WARN 0.65-0.85
    FAIL < 0.65

- Critical row drop fraction:
    PASS <= 0.05
    WARN 0.05-0.15
    FAIL > 0.15

- Numeric inf count:
    PASS only if == 0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(value: Any) -> str:
    try:
        if value is None:
            return "N/A"
        value = float(value)
        if not math.isfinite(value):
            return "N/A"
        return f"{value * 100:.2f}%"
    except Exception:
        return "N/A"


def status_ge(value: float | None, warn_threshold: float, pass_threshold: float) -> str:
    if value is None or not math.isfinite(value):
        return FAIL
    if value >= pass_threshold:
        return PASS
    if value >= warn_threshold:
        return WARN
    return FAIL


def status_le(value: float | None, pass_threshold: float, fail_threshold: float) -> str:
    if value is None or not math.isfinite(value):
        return FAIL
    if value <= pass_threshold:
        return PASS
    if value <= fail_threshold:
        return WARN
    return FAIL


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_check(name: str, value: str, status: str, note: str = "") -> None:
    print(f"[{status:<4}] {name:<34} {value}")
    if note:
        print(f"       {note}")


def extract_filter_report(report: Dict[str, Any]) -> Dict[str, Any]:
    # Support both naming styles used in earlier scripts.
    return (
        report.get("filter_report")
        or report.get("critical_filter_report")
        or {}
    )


def inspect_quality(
    report: Dict[str, Any],
    *,
    greeks_warn: float,
    greeks_pass: float,
    chain_warn: float,
    chain_pass: float,
    row_drop_pass: float,
    row_drop_fail: float,
    missing_warn: float,
    missing_fail: float,
    top_n_missing: int,
) -> Tuple[List[str], List[str], bool]:

    recommendations: List[str] = []
    warnings: List[str] = []
    failed = False

    print_section("CORE QUALITY CHECKS")

    # ------------------------------------------------------------
    # Greeks availability
    # ------------------------------------------------------------
    greeks = report.get("greeks_available_fraction")
    try:
        greeks_f = float(greeks) if greeks is not None else None
    except Exception:
        greeks_f = None

    status = status_ge(greeks_f, greeks_warn, greeks_pass)
    print_check(
        "Greeks available",
        fmt_pct(greeks_f),
        status,
        f"PASS >= {greeks_pass:.0%}, WARN >= {greeks_warn:.0%}",
    )

    if status == FAIL:
        failed = True
        recommendations.append(
            "Greeks availability is too low. Check option-price validity, spot/strike alignment, "
            "expiry timestamps, risk-free rate assumptions, and IV solver convergence."
        )
    elif status == WARN:
        warnings.append(
            "Greeks availability is moderate. Do not make Greeks mandatory hard filters until coverage improves."
        )

    # ------------------------------------------------------------
    # Chain OI availability
    # ------------------------------------------------------------
    chain = report.get("chain_oi_available_fraction")
    try:
        chain_f = float(chain) if chain is not None else None
    except Exception:
        chain_f = None

    status = status_ge(chain_f, chain_warn, chain_pass)
    print_check(
        "Chain OI available",
        fmt_pct(chain_f),
        status,
        f"PASS >= {chain_pass:.0%}, WARN >= {chain_warn:.0%}",
    )

    if status == FAIL:
        failed = True
        recommendations.append(
            "Chain OI coverage is too low. Verify that enough CE/PE strikes around ATM were downloaded "
            "and that open_interest is present across those contracts."
        )
    elif status == WARN:
        warnings.append(
            "Chain OI coverage is incomplete. Use chain availability flags in the model and avoid treating missing OI as zero."
        )

    # ------------------------------------------------------------
    # Critical-filter row loss
    # ------------------------------------------------------------
    filt = extract_filter_report(report)

    rows_before = (
        filt.get("rows_before")
        if "rows_before" in filt
        else filt.get("rows_before_critical_filter")
    )

    rows_after = (
        filt.get("rows_after")
        if "rows_after" in filt
        else filt.get("rows_after_critical_filter")
    )

    rows_dropped = (
        filt.get("rows_dropped")
        if "rows_dropped" in filt
        else filt.get("rows_dropped_critical")
    )

    drop_fraction = None

    try:
        if rows_before is not None:
            rows_before = int(rows_before)
        if rows_after is not None:
            rows_after = int(rows_after)
        if rows_dropped is not None:
            rows_dropped = int(rows_dropped)

        if rows_dropped is None and rows_before is not None and rows_after is not None:
            rows_dropped = rows_before - rows_after

        if rows_before and rows_before > 0 and rows_dropped is not None:
            drop_fraction = rows_dropped / rows_before
    except Exception:
        pass

    status = status_le(drop_fraction, row_drop_pass, row_drop_fail)

    print_check(
        "Critical row drop fraction",
        fmt_pct(drop_fraction),
        status,
        f"Rows before={rows_before}, after={rows_after}, dropped={rows_dropped}",
    )

    if status == FAIL:
        failed = True
        recommendations.append(
            "Too many rows are being discarded by the critical-market filter. "
            "Inspect which critical fields are missing before model training."
        )
    elif status == WARN:
        warnings.append(
            "A noticeable fraction of rows is being dropped. Check whether this is concentrated around expiry changes or illiquid strikes."
        )

    # ------------------------------------------------------------
    # Numeric infinities
    # ------------------------------------------------------------
    inf_count = (
        filt.get("numeric_inf_count")
        if "numeric_inf_count" in filt
        else filt.get("inf_count_after_cleanup")
    )

    try:
        inf_count_i = int(inf_count or 0)
    except Exception:
        inf_count_i = -1

    if inf_count_i == 0:
        status = PASS
    else:
        status = FAIL
        failed = True
        recommendations.append(
            "Numeric infinities remain in the dataset. Replace +/-inf with NaN before training."
        )

    print_check(
        "Numeric +/-inf count",
        str(inf_count_i),
        status,
        "Must be exactly 0.",
    )

    # ------------------------------------------------------------
    # Top missing columns from quality report
    # ------------------------------------------------------------
    top_missing = (
        filt.get("top_missing_columns")
        or filt.get("nan_counts_top_50")
        or {}
    )

    print_section("TOP MISSING COLUMNS")

    if not top_missing:
        print("No top-missing-column summary found in quality report.")
    else:
        # Convert counts to fractions if total row count is known.
        total_rows = rows_after or report.get("option_rows_model_ready") or 0

        items = list(top_missing.items())

        # Sort descending by missing count.
        def sort_key(item):
            try:
                return float(item[1])
            except Exception:
                return -1

        items.sort(key=sort_key, reverse=True)

        for col, count in items[:top_n_missing]:
            try:
                count_i = int(count)
            except Exception:
                count_i = count

            frac = None
            if total_rows:
                try:
                    frac = float(count) / float(total_rows)
                except Exception:
                    pass

            if frac is None:
                status = WARN if count else PASS
                frac_txt = "N/A"
            elif frac >= missing_fail:
                status = FAIL
                failed = True
            elif frac >= missing_warn:
                status = WARN
            else:
                status = PASS

            print(
                f"[{status:<4}] {col:<42} "
                f"missing={count_i!s:<10} "
                f"fraction={fmt_pct(frac)}"
            )

    # ------------------------------------------------------------
    # Dataset summary
    # ------------------------------------------------------------
    print_section("DATASET SUMMARY")

    for key in (
        "underlying_rows",
        "option_rows_full",
        "option_rows_model_ready",
        "option_contracts",
        "contracts",
        "date_start",
        "date_end",
        "start",
        "end",
    ):
        if key in report:
            print(f"{key:<30}: {report[key]}")

    return recommendations, warnings, failed


def inspect_manifest(
    manifest: Dict[str, Any],
    *,
    feature_missing_warn: float,
    feature_missing_fail: float,
    max_rows: int,
) -> bool:

    print_section("FEATURE MANIFEST MISSINGNESS")

    rows = []

    for feature, meta in manifest.items():
        if not isinstance(meta, dict):
            continue

        role = meta.get("role", "feature")
        if role != "feature":
            continue

        missing = meta.get("missing_fraction")

        try:
            missing_f = float(missing)
        except Exception:
            continue

        if missing_f >= feature_missing_warn:
            rows.append(
                (
                    feature,
                    missing_f,
                    meta.get("dtype"),
                    meta.get("unique_non_null"),
                )
            )

    rows.sort(key=lambda x: x[1], reverse=True)

    failed = False

    if not rows:
        print(
            f"No model features have missing_fraction >= {feature_missing_warn:.0%}."
        )
        return False

    for feature, missing, dtype, unique in rows[:max_rows]:
        if missing >= feature_missing_fail:
            status = FAIL
            failed = True
        else:
            status = WARN

        print(
            f"[{status:<4}] {feature:<42} "
            f"missing={missing*100:6.2f}% "
            f"dtype={str(dtype):<12} "
            f"unique={unique}"
        )

    return failed


def print_recommendations(
    recommendations: List[str],
    warnings: List[str],
) -> None:

    print_section("RECOMMENDATIONS")

    if not recommendations and not warnings:
        print("Dataset quality looks healthy enough to proceed to candidate-generation/backtesting.")
        return

    for w in warnings:
        print(f"[WARN] {w}")

    for r in recommendations:
        print(f"[ACTION] {r}")


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--quality-report",
        default="data/banknifty_processed_stream/quality_report.json",
    )

    ap.add_argument(
        "--feature-manifest",
        default="data/banknifty_processed_stream/feature_manifest.json",
    )

    ap.add_argument(
        "--skip-manifest",
        action="store_true",
    )

    # Greeks thresholds
    ap.add_argument("--greeks-warn", type=float, default=0.60)
    ap.add_argument("--greeks-pass", type=float, default=0.80)

    # Chain OI thresholds
    ap.add_argument("--chain-warn", type=float, default=0.65)
    ap.add_argument("--chain-pass", type=float, default=0.85)

    # Critical row drop thresholds
    ap.add_argument("--row-drop-pass", type=float, default=0.05)
    ap.add_argument("--row-drop-fail", type=float, default=0.15)

    # Missing-column thresholds inside quality report
    ap.add_argument("--missing-warn", type=float, default=0.20)
    ap.add_argument("--missing-fail", type=float, default=0.50)

    # Feature manifest thresholds
    ap.add_argument("--feature-missing-warn", type=float, default=0.20)
    ap.add_argument("--feature-missing-fail", type=float, default=0.50)

    ap.add_argument("--top-missing", type=int, default=25)
    ap.add_argument("--manifest-max-rows", type=int, default=40)

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    quality_path = Path(args.quality_report)

    try:
        report = load_json(quality_path)
    except Exception as exc:
        print(f"ERROR: Could not read quality report: {exc}", file=sys.stderr)
        return 2

    recommendations, warnings, failed = inspect_quality(
        report,
        greeks_warn=args.greeks_warn,
        greeks_pass=args.greeks_pass,
        chain_warn=args.chain_warn,
        chain_pass=args.chain_pass,
        row_drop_pass=args.row_drop_pass,
        row_drop_fail=args.row_drop_fail,
        missing_warn=args.missing_warn,
        missing_fail=args.missing_fail,
        top_n_missing=args.top_missing,
    )

    if not args.skip_manifest:
        manifest_path = Path(args.feature_manifest)

        if manifest_path.exists():
            try:
                manifest = load_json(manifest_path)
                manifest_failed = inspect_manifest(
                    manifest,
                    feature_missing_warn=args.feature_missing_warn,
                    feature_missing_fail=args.feature_missing_fail,
                    max_rows=args.manifest_max_rows,
                )

                if manifest_failed:
                    failed = True
                    recommendations.append(
                        "One or more model features have very high missingness. "
                        "Consider excluding them, recalculating them, or using them only with explicit availability flags."
                    )

            except Exception as exc:
                warnings.append(
                    f"Could not inspect feature manifest: {exc}"
                )

        else:
            warnings.append(
                f"Feature manifest not found: {manifest_path}"
            )

    print_recommendations(recommendations, warnings)

    print_section("FINAL STATUS")

    if failed:
        print("FAIL")
        print(
            "Do not move directly to strategy/RL training until the failed data-quality checks are investigated."
        )
        return 1

    if warnings:
        print("WARN")
        print(
            "Dataset is usable for investigation, but review the warnings before making features mandatory."
        )
        return 0

    print("PASS")
    print(
        "Dataset quality checks passed. Proceed to candidate generation and deterministic backtesting."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
