---
name: ofd1-marimo-research
description: >-
  Guides editing, running, and validating the reactive Marimo research notebooks
  (phase1_feasibility.py and phase2_forecasting_trials.py). Activate when the
  user asks to update hydrological analysis, add plots to the notebooks, run
  Marimo in edit/headless mode, or regenerate Phase 2 ablation benchmark files.
---

# OFD1 Marimo Research Notebooks Skill

The empirical hydrological study and model ablation benchmarks are authored as reactive **Marimo** notebooks (`marimo>=0.25.2`).

## 1. Notebook Overview

| Notebook File | Purpose | Key Inputs / Outputs |
| :--- | :--- | :--- |
| `phase1_feasibility.py` | Phase 1 Predictability & Telemetry Data Quality Audit: ThingSpeak pagination (`178,891` packets), June 2024 clock jump analysis, The Stump Oct 2026 stuck counter isolation, high-wind under-catch analysis, CCF (`+2h` lag peak), regime-conditioned ACF, and "Dry Rain" peat bog saturation analysis. | Reads/writes `data/ofd1_thingspeak_raw.parquet`, `data/ofd1_aligned_15m.parquet`, `data/ofd1_aligned_hourly.parquet`. |
| `phase2_forecasting_trials.py` | Phase 2 Out-of-Sample Forecasting Trials (`2026` test set) across `+1h`, `+2h`, `+3h`, `+6h` and extended `+1h..+48h` horizons comparing `0. Naive Persistence`, `1. Linear ARX`, `2. Raw GBDT`, `3. Hydro-GBDT (Causal)`, and `4. Hydro-GBDT (+Future Rain)`. | Reads `data/phase2_trial_metrics.parquet`, `data/phase2_trial_predictions.parquet`, `data/horizon_breakdown_1h_to_48h.parquet`. |

## 2. Running & Checking Marimo Notebooks

1. **Interactive Edit Server:**
   ```bash
   uv run marimo edit phase1_feasibility.py
   uv run marimo edit phase2_forecasting_trials.py
   ```
2. **Syntax & DAG Validation (No Duplicate Cell Definitions):**
   - Marimo enforces a strict directed acyclic graph (DAG) across cells: **no global variable may be defined by more than one `@app.cell`**.
   - Local helper variables inside a cell that should not be exported to the reactive graph must be prefixed with an underscore (e.g., `_sub`, `_fig`, `_check_df`).
   - Verify a notebook parses cleanly and has no DAG cycles or multiple-definition errors by running:
     ```bash
     uv run marimo check phase1_feasibility.py
     uv run marimo check phase2_forecasting_trials.py
     ```
3. **Executing / Exporting Headlessly:**
   ```bash
   uv run python phase2_forecasting_trials.py
   ```
