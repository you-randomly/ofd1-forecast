"""Backfill the archived multi-model NWP rainfall dataset used for verification and backtesting.

Writes ``data/ofd1_nwp_archive.parquet`` (hourly, UTC index) with, for each NWP model slug:

* ``{slug}_d0`` – short-range rain (Open-Meteo Previous Runs ``precipitation``, ~0–6h lead)
* ``{slug}_d1`` – rain predicted 24h before valid time (``precipitation_previous_day1``)
* ``{slug}_d2`` – rain predicted 48h before valid time (``precipitation_previous_day2``)

plus ``pop_short_range_pct`` – short-range precipitation probability from the Historical Forecast
API (stitched first hours of each run; this is *not* what a +12h/+24h forecast knew at issue time
and is only used for descriptive context on the dashboard).

The file is kept separate from ``ofd1_aligned_hourly.parquet`` so it seeds cleanly onto an existing
Kubernetes PVC (seed files are only copied when absent) and is never overwritten by live sync.

Usage::

    uv run python scripts/backfill_nwp_archive.py                # 2024-01-01 → today
    uv run python scripts/backfill_nwp_archive.py --start 2026-01-01
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

LAT, LON = 51.8258, -3.6611
MODELS = [
    "best_match",
    "ukmo_seamless",
    "ecmwf_ifs025",
    "icon_seamless",
    "gfs_seamless",
    "meteofrance_seamless",
]
PREV_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
HIST_FC_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OUT_PATH = Path("data/ofd1_nwp_archive.parquet")


def _chunks(start: date, end: date, days: int = 120):
    cur = start
    while cur <= end:
        nxt = min(end, cur + timedelta(days=days - 1))
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def _get(client: httpx.Client, url: str, params: dict) -> dict:
    for attempt in range(4):
        resp = client.get(url, params=params)
        if resp.status_code == 200:
            return resp.json()
        time.sleep(2.0 * (attempt + 1))
    resp.raise_for_status()
    return {}


def fetch_previous_runs(client: httpx.Client, start: date, end: date) -> pd.DataFrame:
    frames = []
    for c0, c1 in _chunks(start, end):
        print(f"  previous-runs {c0} → {c1}")
        js = _get(
            client,
            PREV_RUNS_URL,
            {
                "latitude": LAT,
                "longitude": LON,
                "hourly": "precipitation,precipitation_previous_day1,precipitation_previous_day2",
                "models": ",".join(MODELS),
                "start_date": c0.isoformat(),
                "end_date": c1.isoformat(),
                "timezone": "UTC",
            },
        )
        h = js["hourly"]
        df = pd.DataFrame(index=pd.to_datetime(h["time"], utc=True))
        for slug in MODELS:
            for var, suffix in [
                ("precipitation", "d0"),
                ("precipitation_previous_day1", "d1"),
                ("precipitation_previous_day2", "d2"),
            ]:
                key = f"{var}_{slug}"
                df[f"{slug}_{suffix}"] = (
                    pd.to_numeric(pd.Series(h[key]), errors="coerce").values if key in h else np.nan
                )
        frames.append(df)
    return pd.concat(frames).sort_index()


def fetch_short_range_pop(client: httpx.Client, start: date, end: date) -> pd.Series:
    parts = []
    for c0, c1 in _chunks(start, end, days=240):
        print(f"  historical-forecast PoP {c0} → {c1}")
        js = _get(
            client,
            HIST_FC_URL,
            {
                "latitude": LAT,
                "longitude": LON,
                "hourly": "precipitation_probability",
                "start_date": c0.isoformat(),
                "end_date": c1.isoformat(),
                "timezone": "UTC",
            },
        )
        h = js["hourly"]
        parts.append(
            pd.Series(
                pd.to_numeric(pd.Series(h["precipitation_probability"]), errors="coerce").values,
                index=pd.to_datetime(h["time"], utc=True),
            )
        )
    return pd.concat(parts).sort_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    out = Path(args.out)

    with httpx.Client(timeout=120.0) as client:
        df = fetch_previous_runs(client, start, end)
        df["pop_short_range_pct"] = fetch_short_range_pop(client, start, end).reindex(df.index)

    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")]
    if out.exists():
        old = pd.read_parquet(out)
        df = pd.concat([old[~old.index.isin(df.index)], df]).sort_index()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    cov = {c: int(df[c].notna().sum()) for c in df.columns if c.endswith("_d1") or c.endswith("_d2")}
    print(f"Saved {len(df)} hourly rows to {out} ({df.index.min()} → {df.index.max()})")
    print("Non-null counts:", cov)


if __name__ == "__main__":
    main()
