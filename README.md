# OFD1 Streamway: Phase 1 Predictability & Data Quality Audit

Reactive diagnostic suite ([`phase1_feasibility.py`](phase1_feasibility.py)) evaluating whether surface rainfall and antecedent moisture states contain sufficient causal signal to forecast streamway water levels in **Ogof Ffynnon Ddu 1 (OFD1)** (`ThingSpeak Channel 2574933`, measured downstream of *The Step*).

## Quick Start

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
- `data/forecast_verification_ledger.parquet`: Stored historical forecast verification ledger (`+1h` to `+24h`) with strict training (`< 2026-01-01 UTC`) vs. out-of-sample (`≥ 2026-01-01 UTC`) provenance.
- `data/phase2_trial_metrics.parquet` & `data/horizon_breakdown_1h_to_48h.parquet`: Out-of-sample 2026 forecasting trial metrics and `+1h` to `+48h` horizon breakdown with `P10–P90` uncertainty bands.

## Key Phase 1 Empirical Findings

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
