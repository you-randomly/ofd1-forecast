---
name: ofd1-web-dashboard-and-api
description: >-
  Guides development, debugging, and verification of the OFD1 FastAPI web
  server (app/main.py) and responsive Caver Dashboard frontend
  (app/static/index.html, app/static/app.js, app/static/styles.css). Activate
  when adding or changing API endpoints, UI tabs, Plotly charts, theme handling,
  or caver safety alert banners.
---

# OFD1 Web Dashboard & FastAPI Skill

The web application serves a real-time Caver Safety & Flood Forecasting Dashboard backed by `OFD1ForecastEngine`.

## 1. Running the Dev Server Locally

Start the FastAPI server using `uv`:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Environment variables supported by `app/main.py`:
- `OFD1_DATA_DIR`: Path to read/write Parquet and model files (default: `data`).
- `OFD1_SEED_DIR`: Directory used on startup to seed missing files into `OFD1_DATA_DIR` (default: `seed_data`, used in Kubernetes PVC mounts).
- `OFD1_REFRESH_SECONDS`: Background ThingSpeak + Open-Meteo sync interval in seconds (default: `600`).

## 2. REST API Endpoint Reference (`app/main.py`)

| Method & Path | Query Parameters | Purpose & Response |
| :--- | :--- | :--- |
| `GET /` | None | Serves `app/static/index.html`. |
| `GET /api/health` | None | Kubernetes startup/readiness/liveness probe (`status`, `sync_status`, `last_sync_utc`, `horizons_loaded`). |
| `GET /api/dashboard` | None | Returns `engine.latest_payload` (`current` KPIs, `gauges` health, `history_72h`, `forecast_48h`, `forecast_anchors`). |
| `POST /api/refresh` | None | Triggers immediate `engine.refresh_live_telemetry()` against ThingSpeak + Open-Meteo and updates the verification ledger. |
| `GET /api/simulate` | `rain_6h_mm` (`0..60`), `moisture_pct` (`5..120`), `start_depth_mm` (`380..2000`, optional) | Runs the interactive "What-If" 6-hour storm simulator across dry vs. saturated peat bog regimes. |
| `GET /api/diagnostics` | None | Cached Phase 1 & Phase 2 diagnostics (`ccf`, `acf`, `dry_rain_storms`, `horizon_breakdown`, `ablation_metrics`). |
| `GET /api/verification` | `horizon_h` (`1, 2, 3, 6, 12, 24`), `window` (`recent_14d`, `oct_2026_floods`, `aug_2026_dry_rain`, `jan_feb_2026_winter`, `full_2026`, `train_2024_2025`, `full_all_2024_2026`) | Returns historical forecast vs. actual verification time series, scorecard metrics, sensor spans, and train/test split provenance. |

## 3. Frontend Architecture & UI Guidelines (`app/static/`)

1. **Four Interactive Views (`index.html` & `app.js`):**
   - **Live 48h Forecast:** Safety alert banner (`GREEN` / `AMBER` / `RED`), 4 KPI cards, 72h history + 48h `P10/P50/P90` Plotly hydrograph with inverted rainfall bars, gauge health cards, and 48h hourly table.
   - **Historical Verification:** Horizon selector (`+1h` to `+24h`), time-window selector, Train (`< 2026-01-01`) vs. Out-of-Sample Test (`>= 2026-01-01`) provenance banner, predicted vs. actual hydrograph, error scatter, scorecard table, and 40-row audit log.
   - **What-If Storm Simulator:** Sliders for 6h rainfall (`mm`), peat bog saturation (`%`), and starting depth (`mm`) demonstrating the nonlinear "Dry Rain" sponge vs. winter flood response.
   - **Hydrology & Model Audit:** CCF (`+2h` peak), regime-conditioned ACF, Dry Rain storm scatter, and 1h–48h model ablation metrics.
2. **Theme Reactivity (`prefers-color-scheme: dark`):**
   - All Plotly charts must use `getPlotlyTheme()` in `app/static/app.js` and re-render inside the `darkThemeMedia.addEventListener("change", ...)` listener so switching OS light/dark mode updates axes, gridlines, and hover labels immediately.
3. **Verification Step:**
   - Whenever modifying `app/main.py` or `app/engine.py`, verify all endpoints return `200 OK` via `fastapi.testclient.TestClient` or `curl http://127.0.0.1:8000/api/health`.
