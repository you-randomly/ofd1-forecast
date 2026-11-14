# NWP Ensemble Backfill & Backtest Scripts (`scripts`)

## Overview
Offline CLI utilities for backfilling multi-model historical Numerical Weather Prediction (NWP) archives from Open-Meteo and evaluating out-of-sample 2026 multi-model ensemble quantile trajectories against single-model deterministic forecasts.

## Interface
- `[scripts/backfill_nwp_archive.py:LAT]`, `[scripts/backfill_nwp_archive.py:LON]` - Coordinates (`51.825`, `-3.662`) for the OFD1 / Black Mountain catchment.
- `[scripts/backfill_nwp_archive.py:MODELS]` - 5-model NWP suite (`ukmo_seamless`, `ecmwf_ifs025`, `icon_seamless`, `meteofrance_seamless`, `gfs_seamless`).
- `[scripts/backfill_nwp_archive.py:PREV_RUNS_URL]` - Open-Meteo Previous Runs API endpoint (`https://previous-runs-api.open-meteo.com/v1/forecast`).
- `[scripts/backfill_nwp_archive.py:HIST_FC_URL]` - Open-Meteo Historical Forecast API endpoint (`https://historical-forecast-api.open-meteo.com/v1/forecast`).
- `[scripts/backfill_nwp_archive.py:OUT_PATH]` - Default archive output path (`data/ofd1_nwp_archive.parquet`).
- `[scripts/backfill_nwp_archive.py:fetch_previous_runs]` - Fetches Day 0, Day 1, and Day 2 precipitation lead runs for all configured NWP models.
- `[scripts/backfill_nwp_archive.py:fetch_short_range_pop]` - Fetches hourly `precipitation_probability` from the Historical Forecast API.
- `[scripts/backfill_nwp_archive.py:main]` - CLI entry point accepting `--start` (`[scripts/backfill_nwp_archive.py:--start]`), `--end` (`[scripts/backfill_nwp_archive.py:--end]`), and `--out` (`[scripts/backfill_nwp_archive.py:--out]`).
- `[scripts/backtest_nwp_ensemble.py:MODELS]` - 5-model NWP ensemble configuration used for backtesting.
- `[scripts/backtest_nwp_ensemble.py:EVAL_HORIZONS]` - Evaluated forecast horizons (`[6, 12, 18, 24, 36, 48]` hours).
- `[scripts/backtest_nwp_ensemble.py:summarize]` - Computes per-horizon RMSE, MAE, 80% PI coverage, 80% PI width, and 700 mm peak capture metrics across deterministic, ensemble, and perfect-weather scenarios.
- `[scripts/backtest_nwp_ensemble.py:main]` - CLI entry point executing the 2026 out-of-sample NWP ensemble backtest.

## Invariants
- `[scripts/backfill_nwp_archive.py:main]` Output index is UTC datetime named `'timestamp'` with duplicates deduplicated keeping `'last'`, merged with any existing non-overlapping timestamps from the target parquet file, and sorted monotonically.
- `[scripts/backtest_nwp_ensemble.py:_build_future_rain_trajectories]` Multi-model future rain trajectories splice lead hours `1..6` from `{slug}_d0`, `7..30` from `{slug}_d1`, and `31..48` from `{slug}_d2`; enforce a 2-model wet consensus quorum (`>= 2` members `>= 0.6 mm/h`) on P90; and apply volume-conservation scaling clipped to `[0.55, 1.15]` so `rain_p10 <= rain_p50 <= rain_p90`.
- `[scripts/backtest_nwp_ensemble.py:main]` Backtest issue timestamps are strictly out-of-sample (`TRAIN_SPLIT_UTC <= issue_idx <= min(valid_obs.index.max(), arch.index.max()) - 48h`), and predicted depth quantiles across deterministic, ensemble, and `perfect_wx` scenarios enforce a `350.0` mm physical floor and monotonic ordering (`350.0 <= p10_mm <= p50_mm <= p90_mm`).

## Side Effects
- `[scripts/backfill_nwp_archive.py:_get]` Makes external HTTP GET requests with retry/sleep backoff to Open-Meteo Previous Runs and Historical Forecast APIs (`PREV_RUNS_URL`, `HIST_FC_URL`).
- `[scripts/backfill_nwp_archive.py:main]` Reads existing parquet file if present, creates parent directories, and writes/updates `data/ofd1_nwp_archive.parquet` (or `--out` path); prints progress and coverage counts to stdout.
- `[scripts/backtest_nwp_ensemble.py:main]` Mutates `sys.path`, reads `data/ofd1_aligned_hourly.parquet`, model artifact (`eng.models_path`), and `data/ofd1_nwp_archive.parquet`, mutates pandas display options, writes `data/nwp_ensemble_backtest.parquet` and `data/nwp_ensemble_backtest_summary.csv`, and prints summary table to stdout.

## Verification
- Run: `pytest`
