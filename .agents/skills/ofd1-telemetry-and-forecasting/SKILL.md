---
name: ofd1-telemetry-and-forecasting
description: >-
  Guides development, debugging, and retraining of the OFD1 hydrological
  telemetry pipeline, glitch filters, peat-tank feature engineering,
  multi-horizon P10/P50/P90 HistGradientBoosting models, and verification
  ledger in app/engine.py. Activate when modifying feature columns, gauge
  failover, model training, what-if storm simulation, or verification metrics.
---

# OFD1 Telemetry & Forecasting Engine Skill

The core hydrological pipeline and ML inference engine live in `OFD1ForecastEngine` (`app/engine.py`).

## 1. Feature Engineering & Glitch Filtering Rules

When modifying `_engineer_features()` or `_attach_horizon_future_cols()` in `app/engine.py`, preserve these physical invariants:

1. **The Stump Stuck-Counter Filter (`stump_glitch_flag`):**
   - Between `2026-10-02 02:00 UTC` and `2026-10-06 09:00 UTC`, `stump_rain_mm` (`field4`) latched at `~96.6 mm/h`.
   - Always mask readings where `stump_rain_mm >= 15.0` while both `swcc_rain_mm < 5.0` and `rain_mm < 5.0`.
2. **Multi-Gauge Fusion & Wind Under-Catch Adjustment:**
   - `club_rain_mean` averages `[stump_rain_clean, swcc_rain_mm]` and falls back to Open-Meteo `rain_mm` (`fused_rain_mm`) when club gauges cut out (e.g., SWCC HQ offline after `2026-11-04`).
   - Apply tipping-bucket wind under-catch scaling: `wind_factor = 1.0 + 0.004 * (wind_kmh - 15.0).clip(lower=0.0)`.
3. **30-Day Rolling Baseflow Datum (`baseflow_30d`):**
   - Computed as the rolling `720h` `5th-percentile` of `depth_mm` to isolate `stage_above_base` across sensor recalibrations.
4. **Evapotranspiration-Adjusted Leaky-Bucket Soil Tanks:**
   - `net_rain = (fused_rain_mm - 0.8 * et0_mm).clip(lower=0.0)`
   - Half-lives `hl in [3, 6, 12, 24, 96, 168]` hours capture both fast epikarst conduits (`3–24h`) and slow blanket peat bog saturation (`96–168h`).
   - `moisture_index` blends normalized ERA5 shallow soil moisture (`0.5`) and `tank_168h` (`0.5`), scaling effective rainfall via `eff_rain_mm = fused_rain_wind_adj * (moisture_index ** 1.5)`.

For the complete list of `CAUSAL_COLS` and `nwp_cols_for_horizon(h)`, read:
- [references/feature-engineering-spec.md](./references/feature-engineering-spec.md)

## 2. Retraining Models Safely (Zero Data Leakage)

If you modify `CAUSAL_COLS`, `nwp_cols_for_horizon(h)`, or model hyperparameters:

1. **Enforce Strict Chronological Split:**
   - `TRAIN_SPLIT_UTC = pd.Timestamp("2026-01-01 00:00:00+00:00")`
   - Filter training rows using **target arrival time**, not issuance time:
     ```python
     target_ts = work.index + pd.Timedelta(hours=h)
     train_slice = work[target_ts < TRAIN_SPLIT_UTC]
     ```
2. **Sample Weighting:**
   - Weight modern multi-gauge era rows (`valid.index >= 2024-05-17 UTC`) by `2.5x` and historical `2015–2023` rows by `1.0x`.
3. **Rebuild Models & Verification Ledger:**
   - Delete or overwrite `data/ofd1_models_v1.joblib` and `data/forecast_verification_ledger.parquet`, then run:
     ```bash
     uv run python -c "from app.engine import OFD1ForecastEngine; e = OFD1ForecastEngine(); e.initialize()"
     ```
4. **Verify Monotonic Quantiles:**
   - Predictions must always enforce non-crossing quantiles:
     - `p50 = max(350.0, base_depth + d_p50)`
     - `p10 = max(350.0, base_depth + min(d_p10, d_p50))`
     - `p90 = max(p50, base_depth + max(d_p90, d_p50))`
