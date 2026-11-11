---
name: ofd1-project-onboarding
description: >-
  Onboards a new agent session to the OFD1 Streamway Predictability & Flood
  Forecasting project (ofd1-phase1). Activate this skill at the start of a new
  session, when asked to explain or explore the repository structure, or before
  making cross-cutting changes across the FastAPI app, Marimo notebooks, data
  pipeline, or Kubernetes manifests.
---

# OFD1 Project Onboarding (`ofd1-phase1`)

This project provides both an empirical hydrological predictability audit and a real-time **48-hour karst streamway flood forecasting & caver safety system** for **Ogof Ffynnon Ddu 1 (OFD1)** in South Wales (`ThingSpeak Channel 2574933`, measured downstream of *The Step*).

## 1. Quick Orientation & Workspace Verification

Run the onboarding verification script first to check Python/uv status, cached Parquet datasets, and pre-trained model weights:

```bash
bash .agents/skills/ofd1-project-onboarding/scripts/verify_workspace.sh
```

> **Note on `uv run` & Virtualenv Paths:** `.venv/bin/python` is symlinked to the user's `uv`-managed Python installation in `~/.local/share/uv/python/cpython-3.13.11-macos-aarch64-none/bin/python3.13`. If `uv run ...` or `.venv/bin/python` fails in Standard Sandbox Mode with `Operation not permitted (os error 1)`, re-run the Python/uv command with `BypassSandbox: true`.

## 2. Repository Map

| Path | Role | Related Skill |
| :--- | :--- | :--- |
| `app/engine.py` | `OFD1ForecastEngine`: telemetry sync (ThingSpeak + Open-Meteo), glitch filtering, peat-tank feature engineering, multi-horizon (`+1h..+48h`) `P10/P50/P90` `HistGradientBoostingRegressor` models, what-if storm simulator, and verification ledger. | `ofd1-telemetry-and-forecasting` |
| `app/main.py` | FastAPI server with background 10-minute telemetry sync loop and REST API endpoints (`/api/health`, `/api/dashboard`, `/api/refresh`, `/api/simulate`, `/api/diagnostics`, `/api/verification`). | `ofd1-web-dashboard-and-api` |
| `app/static/` | Vanilla JS + Plotly + CSS responsive Caver Dashboard (`index.html`, `app.js`, `styles.css`) supporting OS light/dark themes. | `ofd1-web-dashboard-and-api` |
| `phase1_feasibility.py` | Reactive Marimo notebook auditing raw ThingSpeak telemetry, clock jumps, The Stump Oct 2026 stuck counter, CCF (`+2h` time of concentration), ACF, and the "Dry Rain" peat bog saturation anomaly. | `ofd1-marimo-research` |
| `phase2_forecasting_trials.py` | Reactive Marimo notebook benchmarking the 5-step model ablation ladder (`Naive Persistence`, `Linear ARX`, `Raw GBDT`, `Hydro-GBDT Causal`, `Hydro-GBDT +Future Rain`) on the unseen 2026 test split. | `ofd1-marimo-research` |
| `data/` | Cached telemetry parquets (`ofd1_aligned_hourly.parquet`, `ofd1_thingspeak_raw.parquet`), historical archive (`2015–2023`), serialized models (`ofd1_models_v1.joblib`), and verification ledger (`forecast_verification_ledger.parquet`). | `ofd1-telemetry-and-forecasting` |
| `k8s/` & `Dockerfile` | Container image build (`uv` + Python 3.13 non-root `caver` user), PVC seeding logic, `kind` 3-node config, Kustomize manifests, and FluxCD manifests (`k8s/flux-cd/`). | `ofd1-k8s-and-gitops` |

## 3. Core Domain Invariants (Must Not Be Violated)

1. **Caver Safety Thresholds (Downstream of *The Step*):**
   - `STEP_DATUM_MM = 550.0` (`0"` over The Step datum)
   - `AMBER_DEPTH_MM = 620.0` (~`2.8"` over The Step — elevated water / caution)
   - `RED_DEPTH_MM = 700.0` (~`6.0"` over The Step — active flood pulse / unsafe streamway passage)
2. **Karst Time of Concentration ($T_c = +2\text{ hours}$):**
   - Hourly surface rainfall cross-correlation against $\Delta \text{depth}_{t+k}$ peaks sharply at **`+2 hours`** (`r = 0.513` at The Stump, `r = 0.447` at SWCC HQ).
3. **Strict Chronological Train/Test Split (`2026-01-01 00:00:00 UTC`):**
   - **Training Split:** `target_timestamp < 2026-01-01 00:00:00+00:00` (`2015–2025`, with `2024–2025` multi-gauge era weighted `2.5x`).
   - **Out-of-Sample Test Split:** `target_timestamp >= 2026-01-01 00:00:00+00:00` (`2026` unseen evaluation).
   - Never leak `>= 2026-01-01 UTC` target arrivals into model training.
4. **Hardware Glitch Isolation & Gauge Failover:**
   - **The Stump Oct 2026 Stuck Counter:** `field4` latched at `48.2–48.4 mm / 30min` (`96.6 mm/h`) between `2026-10-02 02:00 UTC` and `2026-10-06 09:00 UTC`. Always mask `stump_rain_mm >= 15.0` when `swcc_rain_mm < 5.0` and `rain_mm < 5.0`.
   - **SWCC HQ Cutout:** `field2/3` went offline on `2026-11-04`; `fused_rain_mm` automatically fails over across `[stump_rain_clean, swcc_rain_mm]` mean and Open-Meteo `rain_mm`.

## 4. Detailed Data Dictionary

For full schema details of all Parquet/CSV files in `data/` and external API coordinates, read:
- [references/domain-and-data-dictionary.md](./references/domain-and-data-dictionary.md)
