# OFD1 Streamway: Phase 1 Predictability & Data Quality Audit (`ofd1-phase1` Blueprint)

## System Overview
Reactive diagnostic suite ([`phase1_feasibility.py`](./phase1_feasibility.py), [`phase2_forecasting_trials.py`](./phase2_forecasting_trials.py)) and operational FastAPI forecasting engine ([`app/ README`](./app/README.md)) evaluating and predicting streamway water levels in **Ogof Ffynnon Ddu 1 (OFD1)** (`ThingSpeak Channel 2574933`, measured downstream of *The Step*) using surface rainfall telemetry, antecedent peat-bog saturation state, and a 6-model Open-Meteo Numerical Weather Prediction (NWP) ensemble.

### Quick Start

```bash
# 1. Run the FastAPI Caver Dashboard Web App locally (http://127.0.0.1:8000)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. Deploy via FluxCD (you-randomly/flux-cd)
# Copy k8s/flux-cd/ofd1-forecast.yaml into clusters/my-cluster/apps/ofd1-forecast/ofd1-forecast.yaml
# and add "- ofd1-forecast/ofd1-forecast.yaml" to clusters/my-cluster/apps/kustomization.yaml

# 3. Interactive Research Notebooks (Marimo)
uv run marimo edit phase1_feasibility.py
uv run marimo edit phase2_forecasting_trials.py
```

Cached telemetry & trial datasets in `data/`:
- `data/ofd1_thingspeak_raw.parquet`: Full multi-year raw logger history (`178,891` packets, May 2024 – Nov 2026) across all 5 ThingSpeak fields (`Depth (mm)`, `SWCC Rainfall (mm)`, `SWCC Pressure (hPa)`, `Stump Rainfall (mm)`, `Stump Pressure (hPa)`).
- `data/historic_ofd_streamway_cleaned.csv` & `data/open_meteo_2015_2023.parquet`: Cleaned 2015–2023 streamway & SWCC rainfall archive (`157,474` 30-min rows) and hourly meteorological reanalysis.
- `data/ofd1_aligned_hourly.parquet`: Unified 2015–2026 1-hour UTC resampled telemetry & reanalysis dataset (`94,848` hours, `74,456` valid depth hours; `2015–2025` training split + `2026` unseen out-of-sample test split).
- `data/forecast_verification_ledger.parquet`: Stored historical forecast verification ledger (`+1h` to `+48h`) with strict training (`< 2026-01-01 UTC`) vs. out-of-sample (`≥ 2026-01-01 UTC`) provenance.
- `data/phase2_trial_metrics.parquet` & `data/horizon_breakdown_1h_to_48h.parquet`: Out-of-sample 2026 forecasting trial metrics and `+1h` to `+48h` horizon breakdown with `P10–P90` uncertainty bands.

### Key Phase 1 Empirical Findings

1. **Sanity Check, Clock Synchronization & Gauge Cross-Check**
   - **Out-of-Order Timestamp Jumps:** Raw ThingSpeak `entry_id` ordering contains **2 backward timestamp jumps** in June 2024 (`2024-06-13` jumping back to `2024-06-06` and `2024-05-17` due to SD-card backfill). Sorting and deduplicating by UTC timestamp prior to resampling restores strict monotonicity.
   - **Logger Cadence & Cutouts:** Streamway depth (`field1`) logs every `10.0 min` (`124,812` packets, 6 cutouts `>24h`); SWCC HQ (`field2/3`) logs every `30.02 min` (`34,842` packets, cut out `2026-11-04`); The Stump (`field4/5`) logs every `29.98 min` (`19,225` packets since `2025-05-05`).
   - **The Stump Oct 2026 Stuck-Counter Glitch:** Between `2026-10-02 02:00 UTC` and `2026-10-06 09:00 UTC` (104 hours / 206 packets), The Stump rain counter latched at `48.2–48.4 mm` per 30-minute interval (`96.6 mm/h`), destroying raw inter-gauge correlation (`r = 0.076`). Filtering stuck-counter readings restores SWCC HQ vs. The Stump hourly correlation to **`r = 0.848`** (`r = 0.785` daily against calibrated reference).
   - **High-Wind Tipping-Bucket Under-Catch:** During high winds (`≥25 km/h` vs. `<25 km/h`), tipping-bucket catch efficiency drops by **`11.3%`** at SWCC HQ (`0.604 → 0.536`) and **`6.9%`** at The Stump (`0.639 → 0.595`).

2. **Cross-Correlation Function (CCF) Analysis (`+2h` Time of Concentration)**
   - Lagged cross-correlation between hourly surface rainfall and $\Delta \text{depth}_{t+k}$ peaks sharply at **Lag `+2 hours`**:
     - **The Stump (cleaned):** `r = 0.513` at `+2h` (`r = 0.423` at `+1h`, `r = 0.375` at `+3h`)
     - **SWCC HQ:** `r = 0.447` at `+2h` (`r = 0.387` at `+1h`, `r = 0.303` at `+3h`)
     - **Regional Reference:** `r = 0.403` at `+1h` (`r = 0.347` at `+2h`)
   - Cross-correlation turns negative for `k ≥ 6h` (`r ≈ -0.15` at `+8–10h`), capturing rapid karst conduit drainage after the flood peak passes.

3. **Autocorrelation Function (ACF) of Streamway Depth**
   - **Recession Limb ($\Delta \text{depth} < -2\text{ mm/h}$, dry):** Very high, smooth persistence (`r = 0.998` at `+1h`, `r = 0.993` at `+2h`, `r = 0.926` at `+6h`).
   - **Rising Limb / Flood Pulse ($\Delta \text{depth} > +15\text{ mm/h}$):** Persistence breaks down rapidly (`r = 0.882` at `+2h`, `r = 0.789` at `+3h`, `r = 0.562` at `+6h`, `r = 0.364` at `+12h`), confirming that autoregressive cave level history alone cannot forecast flood pulses without surface rainfall and soil saturation inputs.

4. **Evaluating the "Dry Rain" Anomaly (Antecedent Peat Bog Saturation)**
   - For moderate 6-hour storms (`10–20 mm` on SWCC HQ):
     - **Dry Peat Bog (`moisture_index < 0.35`, e.g., August):** Median streamway rise is only **`+46 mm`** (median peak depth **`516 mm`**, **`0%`** exceeding the `700 mm` amber/red threshold over *The Step*).
     - **Saturated Peat Bog (`moisture_index ≥ 0.65`, e.g., December):** Median streamway rise is **`+605 mm`** (median peak depth **`1,359 mm`**, **`100%`** exceeding `700 mm`).
   - Weighting 6-hour storm rainfall by the antecedent moisture index (combining an evapotranspiration-adjusted EWMA peat tank and shallow soil moisture) tightens the peak depth response correlation from **`r = 0.578` ($R^2 = 0.33$)** to **`r = 0.815` ($R^2 = 0.66$)** and peak rise correlation from **`r = 0.717` to `r = 0.811`**, proving that the OFD1 system is highly forecastable.

## Local Interface & Invariants
- `[pyproject.toml:project.name]`, `[pyproject.toml:project.version]`, `[pyproject.toml:project.requires-python]`, `[pyproject.toml:project.dependencies]`, `[pyproject.toml:dependency-groups.dev]` - Package metadata requiring Python `>=3.13` and dependencies (`fastapi`, `httpx`, `marimo`, `numpy`, `pandas`, `plotly`, `polars`, `pyarrow`, `scikit-learn`, `scipy`, `pytest`, `ruff`).
- `[main.py:main]` - CLI entry point stub printing package greeting.
- `[phase1_feasibility.py:app]` - Reactive Marimo notebook (`[phase1_feasibility.py:__generated_with]`) performing raw ThingSpeak + Open-Meteo data ingestion, glitch filtering, CCF/ACF analysis, and antecedent peat-tank modeling.
  - `[phase1_feasibility.py:need_rebuild]` Rebuilds local Parquet caches if `data/ofd1_aligned_hourly.parquet` or `data/ofd1_thingspeak_raw.parquet` is missing, lacks `'swcc_rain_mm'`, or has fewer than `10,000` rows.
  - `[phase1_feasibility.py:df_raw]` Deduplicates raw ThingSpeak feeds by `'entry_id'`, sorts by `'entry_id'` and UTC `'timestamp'`, and resamples onto uniform 15-minute (interpolation `limit=4`) and 1-hour (interpolation `limit=3`) UTC grids.
  - `[phase1_feasibility.py:stump_rain_clean]` Masks The Stump hourly rainfall values `>= 15.0 mm/h` as `NaN` to filter stuck-counter hardware glitches.
  - `[phase1_feasibility.py:net_rain]` Clips evapotranspiration-adjusted net precipitation (`rain_mm - 0.8 * et0`) at `0.0` and shifts the slow tank EWMA by 1 hour (`shift(1)`) to preserve causality; `[phase1_feasibility.py:sat_norm]` clips normalized ERA5 prior soil moisture to `[0.05, 1.0]` and normalized slow tank to `[0.05, 1.5]`.
- `[phase2_forecasting_trials.py:app]` - Reactive Marimo notebook (`[phase2_forecasting_trials.py:__generated_with]`) evaluating out-of-sample 2026 forecasting models across `['+1h', '+2h', '+3h', '+6h']` (`[phase2_forecasting_trials.py:horizon_select]`) and `+1h` to `+48h` horizon degradation against `NSE = 0.75` and the `700 mm` Amber/Red depth threshold.
- `[Dockerfile:OFD1_DATA_DIR]`, `[Dockerfile:OFD1_SEED_DIR]`, `[Dockerfile:OFD1_REFRESH_SECONDS]`, `[Dockerfile:EXPOSE 8000]`, `[Dockerfile:CMD]` - Container runtime configuration running as non-root user `caver` (`UID 10001`, `[Dockerfile:USER]`) with `uv sync --frozen --no-dev`.

## Subdirectories & System Boundaries
- **app**: [./app/README.md](./app/README.md) - FastAPI web server (`app/main.py`), hydrological feature engineering and `HistGradientBoostingRegressor` P10/P50/P90 forecasting engine (`app/engine.py`), and static Caver Dashboard frontend ([./app/static/README.md](./app/static/README.md)).
- **scripts**: [./scripts/README.md](./scripts/README.md) - Offline CLI utilities for backfilling multi-model historical NWP archives (`scripts/backfill_nwp_archive.py`) and running out-of-sample 2026 NWP ensemble backtests (`scripts/backtest_nwp_ensemble.py`).

## Global Invariants
- **Temporal Train/Test Split (`[app/engine.py:TRAIN_SPLIT_UTC]`)**: All model training across `app/engine.py` and `scripts/backtest_nwp_ensemble.py` strictly restricts training samples to target arrival timestamps (`issue_timestamp + h hours`) `< 2026-01-01 00:00:00+00:00`; all `>= 2026-01-01 00:00:00+00:00` timestamps are reserved as unseen out-of-sample evaluation data.
- **Streamway Safety Thresholds & Hysteresis (`[app/engine.py:STEP_DATUM_MM]`, `[app/engine.py:AMBER_DEPTH_MM]`, `[app/engine.py:RED_DEPTH_MM]`)**: *The Step* datum is fixed at `550.0 mm`, Amber alert threshold at `620.0 mm`, and Red flood threshold at `700.0 mm` (with rising-limb trigger `705.0 mm` and falling-limb clearance `675.0 mm` in `[app/engine.py:RED_TRIGGER_DEPTH_MM]` / `[app/engine.py:RED_CLEAR_DEPTH_MM]`).
- **Quantile Non-Crossing & Physical Floor (`[app/engine.py:OFD1ForecastEngine._forecast_anchors]`, `[scripts/backtest_nwp_ensemble.py:main]`)**: Forecast depth quantiles enforce a `350.0 mm` physical floor and monotonic ordering (`350.0 <= p10 <= p50 <= p90`) across all horizons (`[1, 2, 3, 6, 12, 18, 24, 36, 48]` hours).
- **The Stump Glitch Filtering (`[app/engine.py:OFD1ForecastEngine._engineer_features]`, `[phase1_feasibility.py:stump_rain_clean]`)**: Latched-counter readings at The Stump (`>= 15.0 mm/h` when reference gauges report `< 5.0 mm/h`) are masked to `NaN` before computing composite catchment rainfall.

## Global Side Effects
- **External Telemetry & Weather APIs**: Performs outbound HTTP/HTTPS GET requests to ThingSpeak (`https://api.thingspeak.com/channels/2574933/feeds.json`) and Open-Meteo Forecast, Ensemble, Archive, Previous Runs, and Historical Forecast endpoints (`api.open-meteo.com`, `archive-api.open-meteo.com`, `previous-runs-api.open-meteo.com`, `historical-forecast-api.open-meteo.com`).
- **Local Parquet & Model Storage (`data/`)**: Reads and writes time-series Parquet/CSV datasets (`ofd1_thingspeak_raw.parquet`, `ofd1_aligned_15m.parquet`, `ofd1_aligned_hourly.parquet`, `ofd1_nwp_archive.parquet`, `ofd1_future_nwp_ensemble.parquet`, `forecast_verification_ledger.parquet`, `nwp_ensemble_backtest.parquet`, `nwp_ensemble_backtest_summary.csv`) and serialized `joblib` model artifacts (`ofd1_models_v1.joblib`) inside `OFD1_DATA_DIR` (default `data/`).

## System Verification
- Run: `pytest`
