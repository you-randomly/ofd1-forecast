# OFD1 Domain & Data Dictionary Reference

## 1. External Telemetry & Weather Sources

| Source | Endpoint / Coordinates | Fields & Cadence | Notes |
| :--- | :--- | :--- | :--- |
| **ThingSpeak Channel `2574933`** | `https://api.thingspeak.com/channels/2574933/feeds.json` | `field1`: OFD1 Streamway Depth (`mm`, 10-min cadence)<br>`field2`: SWCC HQ Rainfall (`mm`, 30-min cadence)<br>`field3`: SWCC HQ Pressure (`hPa`, 30-min cadence)<br>`field4`: The Stump Rainfall (`mm`, 30-min cadence, started May 2025)<br>`field5`: The Stump Pressure (`hPa`, 30-min cadence) | Contains 2 backward timestamp jumps in June 2024 due to SD-card backfill; always sort and deduplicate by UTC `timestamp` rather than `entry_id`. |
| **Open-Meteo NWP & ERA5 Reanalysis** | `latitude=51.8258`, `longitude=-3.6611` (Fan Gyhirych / Dan-yr-Ogof upland catchment) | Hourly `precipitation` (`rain_mm`), `temperature_2m` (`temp_c`), `wind_speed_10m` (`wind_kmh`), `surface_pressure` (`era5_press_hpa`), `soil_moisture_0_to_7cm` (`era5_soil_moisture`), `soil_moisture_7_to_28cm` (`era5_soil_deep`), `et0_fao_evapotranspiration` (`et0_mm`). | Provides continuous surface baseline, evapotranspiration for leaky-bucket peat tanks, high-wind under-catch correction, and 48h future NWP forecast. |

---

## 2. Artifacts in `data/`

### Core Telemetry & Archive Files
- **`data/ofd1_aligned_hourly.parquet`** (Tracked in Git / Seeded to PVC):
  - Unified 1-hour UTC index (`timestamp`) from `2015` through `2026` (`~94,848` rows, `~74,456` valid depth rows).
  - Columns: `depth_mm_raw`, `depth_mm`, `swcc_rain_mm`, `stump_rain_mm`, `swcc_press_hpa`, `stump_press_hpa`, `rain_mm`, `temp_c`, `wind_kmh`, `era5_press_hpa`, `era5_soil_moisture`, `era5_soil_deep`, `et0_mm`.
  - Note: Historical `2015–2023` depth values are datum-aligned to the modern sensor baseline (`535.0 mm` baseflow) using a rolling 1440-hour (60-day) 5th-percentile offset.
- **`data/ofd1_thingspeak_raw.parquet`** (Local cache, gitignored):
  - Full raw logger packets (`178,891+` rows, May 2024 – Nov 2026) across `entry_id`, `timestamp`, and all 5 sensor fields.
- **`data/ofd1_aligned_15m.parquet`** (Local cache, gitignored):
  - 15-minute resampled ThingSpeak grid used in `phase1_feasibility.py`.
- **`data/historic_ofd_streamway_cleaned.csv` & `data/open_meteo_2015_2023.parquet`** (Local cache, gitignored):
  - Cleaned 2015–2023 30-minute streamway archive (`157,474` rows) and matching Open-Meteo ERA5 reanalysis.

### Models & Evaluation Artifacts (Tracked in Git / Seeded to PVC)
- **`data/ofd1_models_v1.joblib`**:
  - Nested dictionary `{horizon_h: {"p10": model, "p50": model, "p90": model}}` for `HORIZONS = [1, 2, 3, 6, 12, 18, 24, 36, 48]`.
  - Each model is a `scikit-learn` `HistGradientBoostingRegressor` predicting `target_delta = depth_mm[t+h] - depth_mm[t]`.
- **`data/forecast_verification_ledger.parquet`**:
  - Historical & live forecast verification ledger across `horizon_h in [1, 2, 3, 6, 12, 24]`.
  - Columns: `issued_at`, `target_timestamp`, `horizon_h`, `start_depth_mm`, `pred_p10_mm`, `pred_p50_mm`, `pred_p90_mm`, `actual_depth_mm`, `rain_mm`, `source` (`train_in_sample`, `test_out_of_sample`, `live_forecast`).
- **`data/phase2_trial_metrics.parquet`**:
  - Summary metrics table for the 5-step model ablation ladder across `+1h`, `+2h`, `+3h`, `+6h`.
- **`data/horizon_breakdown_1h_to_48h.parquet`**:
  - Out-of-sample 2026 metrics across all 9 horizons (`+1h` to `+48h`).
