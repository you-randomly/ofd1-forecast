# OFD1 Feature Engineering & Model Specification

## 1. Forecast Horizons & Target Formulation
- **Horizons (`HORIZONS`):** `[1, 2, 3, 6, 12, 18, 24, 36, 48]` hours ahead.
- **Target Variable (`target_delta`):** `depth_mm.shift(-h) - depth_mm` (change in streamway water level in millimeters over lead time `h`).
- **Quantile Regressors per Horizon (`h`):**
  - `p50`: `HistGradientBoostingRegressor(loss="squared_error", max_iter=300, learning_rate=0.04, max_depth=6, min_samples_leaf=25, l2_regularization=2.0, random_state=42)`
  - `p10`: `HistGradientBoostingRegressor(loss="quantile", quantile=0.10, max_iter=220, learning_rate=0.04, max_depth=5, min_samples_leaf=30, random_state=42)`
  - `p90`: `HistGradientBoostingRegressor(loss="quantile", quantile=0.90, max_iter=220, learning_rate=0.04, max_depth=5, min_samples_leaf=30, random_state=42)`

## 2. Strictly Causal Feature Vector (`CAUSAL_COLS` — 63 features)

| Category | Feature Names | Description |
| :--- | :--- | :--- |
| **Hydrograph State & Datum** | `depth_mm`, `stage_above_base`, `baseflow_30d` | Current streamway depth (`mm`), stage above 30-day 5th-percentile baseflow, and rolling 720h baseflow datum. |
| **Hydrograph Velocity & Acceleration** | `ddepth_1h`, `ddepth_2h`, `ddepth_3h`, `ddepth_4h`, `ddepth_6h`, `ddepth_12h`, `ddepth_24h`, `accel_1h` | Depth change over past `1..24h` and 1-hour second difference (`ddepth_1h - ddepth_1h.shift(1)`). |
| **Soil & Peat Saturation** | `moisture_index`, `era5_soil_moisture`, `era5_soil_deep` | Composite peat bog saturation index (`0.05..1.25`), ERA5 `0–7cm` shallow soil moisture, and `7–28cm` deep soil moisture. |
| **Leaky-Bucket EWMA Tanks** | `tank_3h`, `tank_6h`, `tank_12h`, `tank_24h`, `tank_96h`, `tank_168h`, `ref_tank_6h`, `ref_tank_96h` | Exponentially weighted moving averages of `net_rain` (`fused_rain_mm - 0.8 * et0_mm`) and `rain_ref` across epikarst (`3–24h`) and blanket bog (`96–168h`) half-lives. |
| **Meteorology & Barometric** | `wind_kmh`, `temp_c`, `press_hpa`, `dpress_3h`, `dpress_6h` | 10m wind speed, 2m air temperature, fused barometric pressure, and 3h/6h barometric pressure tendency. |
| **Hourly Rain Lags (`0..6h`)** | `fused_rain_lag_{0..6}h`, `ref_rain_lag_{0..6}h`, `eff_rain_lag_{0..6}h` | Wind-adjusted fused gauge rainfall, reference reanalysis rainfall, and moisture-weighted effective rainfall at lags `t-0` through `t-6h`. |
| **Cumulative Rain Windows** | `fused_rain_sum_{2,3,6,12,24}h`, `ref_rain_sum_{2,3,6,12,24}h`, `eff_rain_sum_{2,3,6,12,24}h` | Rolling cumulative rainfall totals over `2h`, `3h`, `6h`, `12h`, and `24h`. |

## 3. Horizon-Coupled NWP Future Features (`nwp_cols_for_horizon(h)`)

To forecast horizons beyond the `+2h` karst time of concentration (`+3h` to `+48h`), the model couples the current causal state with NWP-projected surface rainfall over `[t+1 .. t+h]`:
- `fut_ref_tank_6h`, `fut_ref_tank_24h`, `fut_ref_tank_96h`: Projected tank states at target time `t+h`.
- `fut_rain_sum_6h_at_target`, `fut_rain_sum_12h_at_target`, `fut_eff_rain_sum_6h_at_target`: Rolling 6h/12h rainfall sums ending at target time `t+h`.
- `fut_rain_minus_{0..min(h,6)-1}h_from_target`: Hourly NWP precipitation leading directly into target arrival time `t+h` (especially `t+h-2h`, aligning with the `+2h` karst lag).
