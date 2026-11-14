# OFD1 FastAPI Server & Hydrological Forecast Engine (`app`)

## Overview
FastAPI web application (`app/main.py`) and hydrological telemetry/forecasting engine (`app/engine.py`) for Ogof Ffynnon Ddu 1 (OFD1) streamway flood and water-level forecasting. Synchronizes live ThingSpeak cave/rain telemetry with a 6-model Open-Meteo NWP ensemble, engineers antecedent peat-bog saturation features, trains and evaluates multi-horizon P10/P50/P90 `HistGradientBoostingRegressor` models across `+1h` to `+48h`, maintains an out-of-sample forecast verification ledger, and serves the static Caver Dashboard UI.

## Interface
- `[app/main.py:DATA_DIR]` - Data directory path configured via `OFD1_DATA_DIR` (default `"data"`).
- `[app/main.py:REFRESH_INTERVAL_SECONDS]` - Background telemetry sync interval configured via `OFD1_REFRESH_SECONDS` (default `600`).
- `[app/main.py:STATIC_DIR]` - Path to `./static` frontend assets.
- `[app/main.py:engine]` - Singleton `OFD1ForecastEngine` instance.
- `[app/main.py:lifespan]` - FastAPI async context manager initializing `engine`, warming `_diagnostics_cache`, and managing `_periodic_sync_loop`.
- `[app/main.py:app]` - FastAPI application instance mounting `/static` and REST endpoints.
- `[app/main.py:index]` (`GET /`) - Serves `app/static/index.html`.
- `[app/main.py:health]` (`GET /api/health`) - Returns service health, last telemetry sync timestamp, and sync status.
- `[app/main.py:get_dashboard]` (`GET /api/dashboard`) - Returns the latest 72h observed + 48h forecast dashboard payload.
- `[app/main.py:force_refresh]` (`POST /api/refresh`) - Triggers an immediate `engine.refresh_live_telemetry()` cycle and returns the updated payload.
- `[app/main.py:simulate_storm]` (`GET /api/simulate`) - Runs an interactive synthetic 6-hour storm simulation given `rain_6h_mm`, `moisture_pct`, and optional `start_depth_mm`.
- `[app/main.py:get_diagnostics]` (`GET /api/diagnostics`) - Returns cached Phase 1/2 hydrological diagnostic and horizon NSE series.
- `[app/main.py:get_verification]` (`GET /api/verification`) - Returns historical forecast vs. observed verification metrics, event classifications, and series for `horizon_h` and `window`.
- `[app/main.py:get_forecast_evolution]` (`GET /api/forecast-evolution`) - Returns multi-horizon forecast convergence leading up to `target_timestamp`.
- `[app/engine.py:HORIZONS]` - Forecast lead times `[1, 2, 3, 6, 12, 18, 24, 36, 48]` hours.
- `[app/engine.py:STEP_DATUM_MM]` (`550.0`), `[app/engine.py:AMBER_DEPTH_MM]` (`620.0`), `[app/engine.py:RED_DEPTH_MM]` (`700.0`), `[app/engine.py:RED_TRIGGER_DEPTH_MM]` (`705.0`), `[app/engine.py:RED_CLEAR_DEPTH_MM]` (`675.0`) - Streamway depth safety thresholds and hysteresis bounds in mm.
- `[app/engine.py:TRAIN_SPLIT_UTC]` - Strict training vs. out-of-sample test boundary (`2026-01-01 00:00:00+00:00`).
- `[app/engine.py:CAUSAL_COLS]` - Past/present hydrological feature columns used across all horizons.
- `[app/engine.py:nwp_cols_for_horizon]` - Returns horizon-specific future NWP precipitation feature names for lead `h`.
- `[app/engine.py:OFD1ForecastEngine]` - Core forecasting and telemetry state machine exposing `__init__`, `initialize`, `refresh_live_telemetry`, `simulate_what_if`, `get_diagnostics_summary`, `get_verification_history`, and `get_forecast_evolution_for_target`.
- Subdirectories:
  - [static](./static/README.md) - Single-page Caver Dashboard UI (`index.html`, `app.js`, `styles.css`).

## Invariants
- `[app/main.py:simulate_storm]` Enforces query parameter bounds: `rain_6h_mm` in `[0.0, 60.0]`, `moisture_pct` in `[5.0, 120.0]`, and `start_depth_mm` in `[380.0, 2000.0]` (or `None`).
- `[app/main.py:get_verification]` Coerces `horizon_h` to `2` if it is not in `[1, 2, 3, 6, 12, 18, 24, 36, 48]`.
- `[app/engine.py:OFD1ForecastEngine.initialize]` Raises `FileNotFoundError` if `aligned_1h_path` (`ofd1_aligned_hourly.parquet`) does not exist after seed check; masks historical `depth_raw` outside `[300.0, 3500.0]` mm to `NaN` when merging the 2015–2023 archive.
- `[app/engine.py:OFD1ForecastEngine._engineer_features]` Flags and masks Stump rain glitches where `stump_rain_mm >= 15.0` while both `swcc_rain_mm < 5.0` and `rain_mm < 5.0`; clips `sat_norm` to `[0.05, 1.0]` and `tank_norm` to `[0.05, 1.5]`.
- `[app/engine.py:OFD1ForecastEngine._apply_hydrological_guardrails]` When `base_depth < 620.0` mm and `ddepth_1h < 15.0` mm, deducts `8.0` mm peat sponge retention if `tank_168h < 0.5`, caps `allowed_rise` to `min(25.0 + 12.0 * eff_rain, max(35.0, 30.0 * horizon_h) when eff_rain < 20.0 else 9999.0)`, and enforces `d_p10 <= d_p50 <= allowed_rise` and `d_p90 <= allowed_rise * 1.5 + 20.0`.
- `[app/engine.py:OFD1ForecastEngine._train_and_save_models]` Strictly restricts training samples to target arrival timestamps (`index + h hours`) `< TRAIN_SPLIT_UTC` (`2026-01-01 00:00:00+00:00`), applies `2.5x` sample weight for `index >= 2024-05-17 00:00:00+00:00`, and sets `monotonic_cst = 1` on all `nwp_cols_for_horizon(h)` features.
- `[app/engine.py:OFD1ForecastEngine._enrich_nwp_ensemble_quantiles]` Requires a consensus quorum of `>= 2` NWP models with `>= 0.6 mm/h` before using raw 90th percentile rain, and clips 48h storm volume scaling factor to `[0.55, 1.15]` so `P10 <= P50 <= P90`.
- `[app/engine.py:OFD1ForecastEngine._forecast_anchors]` Enforces physical floor of `350.0` mm and monotonic ordering `350.0 <= p10 <= p50 <= p90_hydro_only <= p90` at every forecast horizon.
- `[app/engine.py:OFD1ForecastEngine._build_dashboard_payload]` Applies hysteresis deadband and persistence debounce for hourly status: rising limb requires `p50_v >= RED_TRIGGER_DEPTH_MM` (`705.0` mm) with adjacent-hour persistence (`>= 700.0` mm) or `base_depth >= AMBER_DEPTH_MM` (`620.0` mm) to enter `RED`, and falling limb stays `RED` while `p50_v >= RED_CLEAR_DEPTH_MM` (`675.0` mm).
- `[app/engine.py:OFD1ForecastEngine.simulate_what_if]` Clips moisture fraction `moist_frac` to `[0.05, 1.30]`, runoff efficiency `runoff_eff` to `[0.12, 1.45]`, and enforces simulated quantile floors and ordering (`p50 >= 380.0`, `p10 >= 360.0`, `p90 >= p50 + 15.0`).
- `[app/engine.py:OFD1ForecastEngine._ensure_and_update_verification_ledger]` Enforces non-crossing quantile predictions with a `350.0` mm floor and deduplicates ledger rows on `(target_timestamp, horizon_h)` keeping `'last'`.
- `[app/engine.py:OFD1ForecastEngine.get_verification_history]` Clips empirical quantile position `q_pct` to `[1, 50]` when `actual_depth_mm <= pred_p50_mm` and `[50, 99]` when `actual_depth_mm > pred_p50_mm`; downsamples plot series with stride `24` (`>15000` rows), `6` (`>4000` rows), `3` (`>1500` rows), or `1`.

## Side Effects
- `[app/main.py:DATA_DIR]` and `[app/main.py:REFRESH_INTERVAL_SECONDS]` Read environment variables `OFD1_DATA_DIR` and `OFD1_REFRESH_SECONDS` at import time; `[app/main.py:engine]` instantiates `OFD1ForecastEngine` (which creates `data_dir` on disk via `mkdir(parents=True, exist_ok=True)`).
- `[app/main.py:lifespan]` Initializes `engine`, populates module-level `_diagnostics_cache`, and spawns background `asyncio` task `_periodic_sync_loop` on FastAPI startup, cancelling it on shutdown.
- `[app/engine.py:OFD1ForecastEngine.initialize]` Reads `OFD1_SEED_DIR` env var and copies missing seed files to `data_dir`; reads/writes Parquet and CSV files in `data_dir` (`ofd1_aligned_hourly.parquet`, `historic_ofd_streamway_cleaned.csv`, `open_meteo_2015_2023.parquet`, `ofd1_nwp_archive.parquet`, `ofd1_future_nwp_ensemble.parquet`); loads or trains/saves `joblib` models at `ofd1_models_v1.joblib`; updates the verification ledger at `forecast_verification_ledger.parquet`.
- `[app/engine.py:OFD1ForecastEngine.refresh_live_telemetry]` Makes outbound HTTPS GET requests via `httpx.AsyncClient` to `api.thingspeak.com` (channel `2574933`) and `api.open-meteo.com` (`/v1/forecast` deterministic and 6-model ensemble), writes updated Parquet files to `self.future_wx_path` and `self.aligned_1h_path`, updates `self.ledger_path`, and mutates in-memory state (`self.df_hourly`, `self.df_future_wx`, `self.df_feat_cached`, `self.latest_payload`, `self.last_sync_utc`, `self.sync_status`, `self._verification_cache`).
- `[app/engine.py:OFD1ForecastEngine.get_verification_history]` and `[app/engine.py:OFD1ForecastEngine.get_forecast_evolution_for_target]` Lazily initialize/update `self.ledger_path` via `_ensure_and_update_verification_ledger` if missing, read Parquet/CSV files from `data_dir`, and populate `self._verification_cache`.

## Verification
- Run: `pytest`
