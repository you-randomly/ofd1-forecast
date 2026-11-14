"""Offline Stage-3 audit & optional NRW telemetered rain-gauge backfill script.

Audits the historical rain_source distribution in data/ofd1_aligned_hourly.parquet
across 2024-2026 and, when NRW_API_KEY is exported in the environment, queries
Natural Resources Wales StationData/byLocation and StationData/Historical (~12.5 months
of 15-minute parameterReadings), aggregates complete hourly buckets (>=3 samples/h),
and merges them into nrw_rain_mm.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.hydrology import (
    NRW_EXCLUDED_STATION_IDS,
    engineer_features,
    extract_nrw_rain_targets,
    parse_nrw_rainfall_payload,
)


def audit_rain_sources(aligned_path: Path) -> None:
    """Print 2024-2026 rain_source distribution in ofd1_aligned_hourly.parquet."""
    df_hourly = pd.read_parquet(aligned_path).sort_index()
    if not isinstance(df_hourly.index, pd.DatetimeIndex):
        df_hourly.index = pd.to_datetime(df_hourly.index, utc=True)

    df_feat = engineer_features(df_hourly)
    modern = df_feat[df_feat.index >= pd.Timestamp("2024-01-01 00:00:00+00:00")]
    counts = modern["rain_source"].value_counts()
    print(f"=== Rain source breakdown 2024-2026 ({len(modern)} hourly rows) ===")
    for src, cnt in counts.items():
        pct = 100.0 * float(cnt) / max(len(modern), 1)
        print(f"  {src:12s}: {int(cnt):6d} h ({pct:5.2f}%)")

    nov_mask = modern.index >= pd.Timestamp("2026-11-01 00:00:00+00:00")
    if nov_mask.any():
        nov_counts = modern.loc[nov_mask, "rain_source"].value_counts()
        print(f"=== Rain source breakdown Nov 2026 ({int(nov_mask.sum())} hourly rows) ===")
        for src, cnt in nov_counts.items():
            pct = 100.0 * float(cnt) / int(nov_mask.sum())
            print(f"  {src:12s}: {int(cnt):6d} h ({pct:5.2f}%)")


def sync_nrw_stations(
    aligned_path: Path,
    lat: float,
    lon: float,
    distance_m: int,
    dry_run: bool,
    include_all_stations: bool = False,
) -> None:
    """Query NRW StationData/byLocation + Historical and optionally persist nrw_rain_mm."""
    api_key = os.environ.get("NRW_API_KEY", "").strip()
    if not api_key:
        print(
            "\nNRW_API_KEY is not set. To enable NRW Stage-3 observed-rain fallback:\n"
            "  1. Register for a free subscription key at https://api-portal.naturalresources.wales/\n"
            "  2. Subscribe to the 'River Level, Rainfall and Sea Data' API\n"
            "  3. Export NRW_API_KEY='<your-key>' and re-run without --dry-run."
        )
        return

    headers = {"Ocp-Apim-Subscription-Key": api_key}
    excluded = set() if include_all_stations else NRW_EXCLUDED_STATION_IDS
    loc_url = (
        "https://api.naturalresources.wales/rivers-and-seas/v1/api/StationData/byLocation"
        f"?distance={distance_m}&lat={lat}&lon={lon}"
    )
    print(f"\nQuerying NRW API within {distance_m} m of ({lat}, {lon})...")
    with httpx.Client(headers=headers, timeout=20.0) as client:
        resp = client.get(loc_url)
        resp.raise_for_status()
        by_loc_json = resp.json()
        targets = extract_nrw_rain_targets(by_loc_json, exclude_station_ids=excluded)
        print(
            f"Selected {len(targets)} NRW rain station(s): "
            f"{[f'{title} (#{loc})' for loc, _, title in targets]}"
        )

        hist_payloads: list[dict] = []
        for loc_id, param_id, title in targets:
            hist_url = (
                "https://api.naturalresources.wales/rivers-and-seas/v1/api/StationData/Historical"
                f"?location={loc_id}&parameter={param_id}"
            )
            h_resp = client.get(hist_url)
            h_resp.raise_for_status()
            h_json = h_resp.json()
            if isinstance(h_json, dict) and h_json.get("parameterReadings"):
                print(f"  Fetched {len(h_json['parameterReadings'])} 15-min readings from {title}")
                hist_payloads.append(h_json)

    if hist_payloads:
        nrw_series, meta = parse_nrw_rainfall_payload(
            hist_payloads, exclude_station_ids=excluded
        )
    else:
        nrw_series, meta = parse_nrw_rainfall_payload(
            by_loc_json, exclude_station_ids=excluded
        )

    if nrw_series.empty:
        print("No valid rainfall readings returned in this query.")
        return

    print(
        f"Parsed {len(nrw_series)} complete hourly points from {meta['station_names']} "
        f"({nrw_series.index.min()} .. {nrw_series.index.max()})"
    )
    if dry_run:
        print("Dry-run mode enabled; parquet file left untouched.")
        return

    df_hourly = pd.read_parquet(aligned_path).sort_index()
    if "nrw_rain_mm" not in df_hourly.columns:
        df_hourly["nrw_rain_mm"] = np.nan
    common = df_hourly.index.intersection(nrw_series.index)
    # Replace rather than preserve stale snapshots on matched stations' span
    df_hourly["nrw_rain_mm"] = nrw_series.reindex(df_hourly.index)
    df_hourly.to_parquet(aligned_path)
    print(f"Updated nrw_rain_mm for {len(common)} overlapping hours in {aligned_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit rain sources and optionally backfill NRW gauges.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--lat", type=float, default=51.8258)
    parser.add_argument("--lon", type=float, default=-3.6611)
    parser.add_argument("--distance-m", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-all-stations", action="store_true")
    args = parser.parse_args()

    aligned_path = args.data_dir / "ofd1_aligned_hourly.parquet"
    audit_rain_sources(aligned_path)
    sync_nrw_stations(
        aligned_path,
        args.lat,
        args.lon,
        args.distance_m,
        args.dry_run,
        include_all_stations=args.include_all_stations,
    )


if __name__ == "__main__":
    main()
