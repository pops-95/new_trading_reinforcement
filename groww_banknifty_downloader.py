#!/usr/bin/env python3
"""
Groww BANKNIFTY historical downloader
======================================

Downloads:
  1. BANKNIFTY underlying 1-minute OHLC
  2. BANKNIFTY nearest-expiry option contracts around ATM (default ATM ±8)
  3. Option 1-minute OHLC + volume + open interest

Default period:
  2024-01-01 through 2026-08-31

Important:
- Uses Groww backtesting methods:
    get_expiries()
    get_contracts()
    get_historical_candles()
- Groww documents a maximum 30-day request span for 1m/2m/3m/5m candles.
- Historical candles include OI for FNO.
- Historical IV/Greeks are not supplied by the historical-candle endpoint.
  The script therefore leaves IV/delta/gamma/theta/vega as NaN for later
  offline calculation.
- Genuine volume=0 and OI=0 are preserved.
- Invalid/non-positive OHLC values are set to NaN.
- No forward filling is performed.
- Downloads are cached/resumable.

Install:
  pip install growwapi pandas numpy pyarrow

PowerShell:
  $env:GROWW_API_TOKEN="YOUR_TOKEN"
  python groww_banknifty_downloader.py

Linux:
  export GROWW_API_TOKEN="YOUR_TOKEN"
  python groww_banknifty_downloader.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from growwapi import GrowwAPI


START_DEFAULT = "2024-01-01"
END_DEFAULT = "2026-08-31"
MARKET_OPEN = "09:15:00"
MARKET_CLOSE = "15:30:00"
MAX_1M_DAYS = 30
STRIKE_STEP_DEFAULT = 100.0
ATM_WINGS_DEFAULT = 8

OPTION_RE = re.compile(
    r"^(?P<exchange>[A-Z]+)-(?P<underlying>[A-Z0-9]+)-"
    r"(?P<expiry>\d{2}[A-Za-z]{3}\d{2})-"
    r"(?P<strike>\d+(?:\.\d+)?)-(?P<option_type>CE|PE)$"
)


@dataclass
class Stats:
    api_calls: int = 0
    retries: int = 0
    cached_skips: int = 0
    underlying_chunks: int = 0
    option_contracts: int = 0
    failed: int = 0
    empty: int = 0


def atomic_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def setup_logging(root: Path, verbose: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(root / "download.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def date_chunks(start: pd.Timestamp, end: pd.Timestamp) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp]]:
    cur = start.normalize()
    end = end.normalize()
    while cur <= end:
        stop = min(cur + pd.Timedelta(days=MAX_1M_DAYS - 1), end)
        yield cur, stop
        cur = stop + pd.Timedelta(days=1)


def month_iter(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1)
    stop = pd.Timestamp(end.year, end.month, 1)
    while cur <= stop:
        yield cur.year, cur.month
        cur += pd.offsets.MonthBegin(1)


def dt_text(day: pd.Timestamp, clock: str) -> str:
    return f"{day:%Y-%m-%d} {clock}"


def response_list(resp: Any, key: str) -> List[Any]:
    if not isinstance(resp, dict):
        return []
    if isinstance(resp.get(key), list):
        return resp[key]
    payload = resp.get("payload")
    if isinstance(payload, dict) and isinstance(payload.get(key), list):
        return payload[key]
    return []


def response_candles(resp: Any) -> List[Any]:
    return response_list(resp, "candles")


def parse_option_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    m = OPTION_RE.match(symbol)
    if not m:
        return None
    d = m.groupdict()
    expiry = pd.to_datetime(d["expiry"], format="%d%b%y", errors="coerce")
    if pd.isna(expiry):
        return None
    return {
        "groww_symbol": symbol,
        "underlying": d["underlying"],
        "expiry_date": expiry.normalize(),
        "strike": float(d["strike"]),
        "option_type": d["option_type"],
    }


def clean_candles(candles: Sequence[Sequence[Any]], symbol: str, fno: bool) -> pd.DataFrame:
    rows = []
    for x in candles:
        if not isinstance(x, (list, tuple)) or len(x) < 5:
            continue
        rows.append({
            "timestamp": x[0],
            "open": x[1] if len(x) > 1 else None,
            "high": x[2] if len(x) > 2 else None,
            "low": x[3] if len(x) > 3 else None,
            "close": x[4] if len(x) > 4 else None,
            "volume": x[5] if len(x) > 5 else None,
            "open_interest": x[6] if fno and len(x) > 6 else None,
            "groww_symbol": symbol,
        })

    cols = ["timestamp", "open", "high", "low", "close", "volume", "open_interest", "groww_symbol"]
    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for c in ["open", "high", "low", "close", "volume", "open_interest"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Price zero/negative is invalid. Volume/OI zero can be genuine.
    for c in ["open", "high", "low", "close"]:
        df.loc[df[c] <= 0, c] = np.nan

    df.loc[df["volume"] < 0, "volume"] = np.nan
    df.loc[df["open_interest"] < 0, "open_interest"] = np.nan

    df = df[df["timestamp"].notna()].copy()

    # OHLC consistency. Do not repair/fill bad values.
    complete = df[["open", "high", "low", "close"]].notna().all(axis=1)
    impossible = complete & (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df["high"] < df["low"])
    )
    df.loc[impossible, ["open", "high", "low", "close"]] = np.nan

    return (
        df.sort_values("timestamp")
        .drop_duplicates(["timestamp", "groww_symbol"], keep="last")
        .reset_index(drop=True)
    )


class Client:
    def __init__(self, token: str, stats: Stats, gap: float, retries: int):
        self.api = GrowwAPI(token)
        self.stats = stats
        self.gap = gap
        self.max_retries = retries
        self.last_call = 0.0

    def call(self, name: str, **kwargs):
        fn = getattr(self.api, name)
        last_exc = None
        for attempt in range(self.max_retries):
            elapsed = time.monotonic() - self.last_call
            if elapsed < self.gap:
                time.sleep(self.gap - elapsed)
            try:
                self.stats.api_calls += 1
                out = fn(**kwargs)
                self.last_call = time.monotonic()
                return out
            except Exception as exc:
                last_exc = exc
                self.stats.retries += 1
                self.last_call = time.monotonic()
                delay = min(60.0, 1.7 ** attempt)
                logging.warning(
                    "%s failed (%d/%d): %s; retry in %.1fs",
                    name, attempt + 1, self.max_retries, exc, delay
                )
                time.sleep(delay)
        self.stats.failed += 1
        raise RuntimeError(f"{name} failed after {self.max_retries} attempts") from last_exc


def fetch_underlying(client: Client, root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    chunk_dir = root / "underlying" / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for a, b in date_chunks(start, end):
        path = chunk_dir / f"BANKNIFTY_1m_{a:%Y%m%d}_{b:%Y%m%d}.parquet"
        if path.exists():
            client.stats.cached_skips += 1
            continue

        logging.info("Underlying %s -> %s", a.date(), b.date())
        resp = client.call(
            "get_historical_candles",
            exchange=client.api.EXCHANGE_NSE,
            segment=client.api.SEGMENT_CASH,
            groww_symbol="NSE-BANKNIFTY",
            start_time=dt_text(a, MARKET_OPEN),
            end_time=dt_text(b, MARKET_CLOSE),
            candle_interval=client.api.CANDLE_INTERVAL_MIN_1,
        )
        candles = response_candles(resp)
        if not candles:
            client.stats.empty += 1
        df = clean_candles(candles, "NSE-BANKNIFTY", False)
        df["instrument_type"] = "INDEX"
        atomic_parquet(df, path)
        client.stats.underlying_chunks += 1

    files = sorted(chunk_dir.glob("BANKNIFTY_1m_*.parquet"))
    if not files:
        return pd.DataFrame()

    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()].copy()
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    day = df["timestamp"].dt.normalize()
    tm = df["timestamp"].dt.time
    open_t = datetime.strptime(MARKET_OPEN, "%H:%M:%S").time()
    close_t = datetime.strptime(MARKET_CLOSE, "%H:%M:%S").time()
    df = df[(day >= start) & (day <= end) & (tm >= open_t) & (tm <= close_t)].reset_index(drop=True)

    atomic_parquet(df, root / "underlying" / "banknifty_1min.parquet")
    atomic_json({
        "rows": len(df),
        "start": None if df.empty else df["timestamp"].min(),
        "end": None if df.empty else df["timestamp"].max(),
        "missing_price_rows": int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum()) if not df.empty else 0,
        "zero_volume_rows": int((df["volume"] == 0).sum()) if not df.empty else 0,
        "duplicates": int(df["timestamp"].duplicated().sum()) if not df.empty else 0,
    }, root / "underlying" / "quality.json")
    return df


def expiries(client: Client, cache: Path, start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    vals = set()
    # Include one month after end so late-August sessions can map to early-Sep expiry if needed.
    search_end = end + pd.offsets.MonthEnd(1)
    for y, m in month_iter(start, search_end):
        path = cache / f"expiries_{y}_{m:02d}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            client.stats.cached_skips += 1
            arr = data.get("expiries", [])
        else:
            resp = client.call(
                "get_expiries",
                exchange=client.api.EXCHANGE_NSE,
                underlying_symbol="BANKNIFTY",
                year=y,
                month=m,
            )
            arr = response_list(resp, "expiries")
            atomic_json({"expiries": arr}, path)

        for x in arr:
            t = pd.to_datetime(x, errors="coerce")
            if pd.notna(t):
                vals.add(t.normalize())
    return sorted(vals)


def contracts(client: Client, cache: Path, expiry: pd.Timestamp) -> List[str]:
    path = cache / f"contracts_{expiry:%Y-%m-%d}.json"
    if path.exists():
        client.stats.cached_skips += 1
        return json.loads(path.read_text(encoding="utf-8")).get("contracts", [])

    resp = client.call(
        "get_contracts",
        exchange=client.api.EXCHANGE_NSE,
        underlying_symbol="BANKNIFTY",
        expiry_date=f"{expiry:%Y-%m-%d}",
    )
    arr = response_list(resp, "contracts")
    atomic_json({"contracts": arr}, path)
    return arr


def session_table(spot: pd.DataFrame, exp: List[pd.Timestamp]) -> pd.DataFrame:
    x = spot.dropna(subset=["timestamp", "close"]).copy()
    x["session_date"] = x["timestamp"].dt.normalize()
    s = x.groupby("session_date", as_index=False).agg(
        session_open=("open", "first"),
        session_close=("close", "last"),
        session_high=("high", "max"),
        session_low=("low", "min"),
    )
    e = pd.DataFrame({"expiry_date": pd.to_datetime(exp)}).sort_values("expiry_date")
    return pd.merge_asof(
        s.sort_values("session_date"),
        e,
        left_on="session_date",
        right_on="expiry_date",
        direction="forward",
        allow_exact_matches=True,
    )


def build_plan(
    sessions: pd.DataFrame,
    contract_map: Dict[pd.Timestamp, List[str]],
    wings: int,
    step: float,
) -> pd.DataFrame:
    parsed: Dict[pd.Timestamp, pd.DataFrame] = {}
    for expiry, symbols in contract_map.items():
        rows = [parse_option_symbol(s) for s in symbols]
        rows = [r for r in rows if r and r["underlying"] == "BANKNIFTY"]
        parsed[expiry] = pd.DataFrame(rows)

    out = []
    for r in sessions.itertuples(index=False):
        if pd.isna(r.expiry_date):
            continue
        expiry = pd.Timestamp(r.expiry_date).normalize()
        c = parsed.get(expiry)
        if c is None or c.empty:
            continue

        # Used only to decide which option files to acquire.
        spot = float(r.session_open) if pd.notna(r.session_open) and r.session_open > 0 else float(r.session_close)
        if not np.isfinite(spot) or spot <= 0:
            continue

        atm = round(spot / step) * step
        wanted = {atm + k * step for k in range(-wings, wings + 1)}
        chosen = c[c["strike"].isin(wanted) & c["option_type"].isin(["CE", "PE"])]

        for q in chosen.itertuples(index=False):
            out.append({
                "session_date": pd.Timestamp(r.session_date),
                "expiry_date": expiry,
                "groww_symbol": q.groww_symbol,
                "strike": q.strike,
                "option_type": q.option_type,
                "spot_reference": spot,
                "atm_reference": atm,
            })

    if not out:
        return pd.DataFrame()
    return (
        pd.DataFrame(out)
        .drop_duplicates(["session_date", "groww_symbol"])
        .sort_values(["session_date", "strike", "option_type"])
        .reset_index(drop=True)
    )


def fetch_option_contract(
    client: Client,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    path: Path,
) -> None:
    if path.exists():
        client.stats.cached_skips += 1
        return

    frames = []
    for a, b in date_chunks(start, end):
        resp = client.call(
            "get_historical_candles",
            exchange=client.api.EXCHANGE_NSE,
            segment=client.api.SEGMENT_FNO,
            groww_symbol=symbol,
            start_time=dt_text(a, MARKET_OPEN),
            end_time=dt_text(b, MARKET_CLOSE),
            candle_interval=client.api.CANDLE_INTERVAL_MIN_1,
        )
        arr = response_candles(resp)
        if not arr:
            client.stats.empty += 1
            continue
        frames.append(clean_candles(arr, symbol, True))

    df = pd.concat(frames, ignore_index=True) if frames else clean_candles([], symbol, True)
    if not df.empty:
        df = df.sort_values("timestamp").drop_duplicates(["timestamp", "groww_symbol"], keep="last")

    meta = parse_option_symbol(symbol) or {}
    df["expiry_date"] = meta.get("expiry_date")
    df["strike"] = meta.get("strike")
    df["option_type"] = meta.get("option_type")
    df["underlying"] = "BANKNIFTY"

    # Explicit nullable fields for the later offline Greeks pipeline.
    for c in ["implied_volatility", "delta", "gamma", "theta", "vega"]:
        df[c] = np.nan

    atomic_parquet(df.reset_index(drop=True), path)
    client.stats.option_contracts += 1


def quality_option(path: Path) -> Dict[str, Any]:
    df = pd.read_parquet(path)
    return {
        "file": str(path),
        "rows": len(df),
        "missing_price_rows": int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum()) if len(df) else 0,
        "zero_volume_rows": int((df["volume"] == 0).sum()) if len(df) else 0,
        "zero_oi_rows": int((df["open_interest"] == 0).sum()) if len(df) else 0,
        "missing_oi_rows": int(df["open_interest"].isna().sum()) if len(df) else 0,
        "duplicates": int(df.duplicated(["timestamp", "groww_symbol"]).sum()) if len(df) else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START_DEFAULT)
    ap.add_argument("--end", default=END_DEFAULT)
    ap.add_argument("--output-dir", default="data/banknifty_groww")
    ap.add_argument("--atm-wings", type=int, default=ATM_WINGS_DEFAULT)
    ap.add_argument("--strike-step", type=float, default=STRIKE_STEP_DEFAULT)
    ap.add_argument("--min-call-gap", type=float, default=0.25)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--underlying-only", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if end < start:
        raise SystemExit("--end must be >= --start")

    root = Path(args.output_dir)
    setup_logging(root, args.verbose)

    token = os.getenv("GROWW_API_TOKEN")
    if not token:
        raise SystemExit(
            "GROWW_API_TOKEN is not set.\n"
            'PowerShell: $env:GROWW_API_TOKEN="YOUR_TOKEN"\n'
            'Linux: export GROWW_API_TOKEN="YOUR_TOKEN"'
        )

    stats = Stats()
    client = Client(token, stats, args.min_call_gap, args.max_retries)

    atomic_json({
        "underlying": "BANKNIFTY",
        "start": start.date(),
        "end": end.date(),
        "interval": "1minute",
        "nearest_expiry_only": True,
        "atm_wings": args.atm_wings,
        "strike_step": args.strike_step,
        "missing_value_policy": {
            "OHLC <= 0": "NaN",
            "volume == 0": "preserve",
            "OI == 0": "preserve",
            "null": "NaN",
            "forward_fill": False,
        },
    }, root / "config.json")

    logging.info("1/5 Fetching BANKNIFTY 1-minute underlying...")
    spot = fetch_underlying(client, root, start, end)
    if spot.empty:
        raise SystemExit("No BANKNIFTY underlying data was returned.")

    if args.underlying_only:
        atomic_json(asdict(stats), root / "stats.json")
        return

    cache = root / "metadata_cache"
    cache.mkdir(parents=True, exist_ok=True)

    logging.info("2/5 Fetching/caching BANKNIFTY expiries...")
    exps = expiries(client, cache, start, end)
    if not exps:
        raise SystemExit("No expiries found.")

    sessions = session_table(spot, exps)
    atomic_parquet(sessions, root / "session_expiry_map.parquet")

    used_exps = sorted(pd.Timestamp(x).normalize() for x in sessions["expiry_date"].dropna().unique())
    logging.info("Nearest expiries used: %d", len(used_exps))

    logging.info("3/5 Fetching/caching contract lists...")
    cmap = {}
    for i, exp in enumerate(used_exps, 1):
        logging.info("Expiry %d/%d: %s", i, len(used_exps), exp.date())
        cmap[exp] = contracts(client, cache, exp)

    logging.info("4/5 Building ATM ±%d strike acquisition plan...", args.atm_wings)
    plan = build_plan(sessions, cmap, args.atm_wings, args.strike_step)
    if plan.empty:
        raise SystemExit("No option contracts matched the acquisition plan.")
    atomic_parquet(plan, root / "option_download_plan.parquet")
    logging.info(
        "Plan rows=%s, unique option contracts=%s",
        f"{len(plan):,}", f"{plan['groww_symbol'].nunique():,}"
    )

    logging.info("5/5 Downloading only required option contracts...")
    options_dir = root / "options" / "contracts"
    options_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    groups = list(plan.groupby("groww_symbol", sort=True))
    for i, (symbol, g) in enumerate(groups, 1):
        first = pd.Timestamp(g["session_date"].min()).normalize()
        expiry = pd.Timestamp(g["expiry_date"].iloc[0]).normalize()
        last = min(pd.Timestamp(g["session_date"].max()).normalize(), expiry, end)
        path = options_dir / f"{symbol}.parquet"

        logging.info("%d/%d %s", i, len(groups), symbol)
        try:
            fetch_option_contract(client, symbol, first, last, path)
            status, error = "ok", None
        except Exception as exc:
            logging.exception("Failed: %s", symbol)
            status, error = "failed", str(exc)

        manifest.append({
            "groww_symbol": symbol,
            "first_needed_session": first,
            "last_needed_session": last,
            "expiry_date": expiry,
            "status": status,
            "error": error,
            "path": str(path),
        })
        atomic_parquet(pd.DataFrame(manifest), root / "option_manifest.parquet")

    q = [quality_option(p) for p in sorted(options_dir.glob("*.parquet"))]
    if q:
        atomic_parquet(pd.DataFrame(q), root / "option_quality.parquet")

    atomic_json(asdict(stats), root / "stats.json")
    logging.info("Finished. Stats: %s", asdict(stats))
    logging.info("Output: %s", root.resolve())


if __name__ == "__main__":
    main()
