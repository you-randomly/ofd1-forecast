# Caver Dashboard Static Frontend (`app/static`)

## Overview
Client-side single-page web dashboard (`index.html`, `app.js`, and `styles.css`) for the OFD1 Streamway Flood & Level Forecast system. Renders interactive Plotly hydrographs, real-time gauge health badges, 48-hour forecast checkpoints, out-of-sample forecast verification diagnostics, interactive storm "What-If" simulations, and Phase 1/2 hydrological ablation charts.

## Interface
- `[app/static/app.js:darkThemeMedia]` - `MediaQueryList` tracking `(prefers-color-scheme: dark)`.
- `[app/static/app.js:isDark]` - Boolean state indicating whether dark theme rendering is active.
- `[app/static/app.js:latestDashboardData]` - Cached payload from `/api/dashboard` or `/api/refresh`.
- `[app/static/app.js:latestDiagnosticsData]` - Cached payload from `/api/diagnostics`.
- `[app/static/app.js:latestSimData]` - Cached payload from `/api/simulate`.
- `[app/static/app.js:latestVerificationData]` - Cached payload from `/api/verification`.
- `[app/static/app.js:getPlotlyTheme]` - Returns theme-aware color tokens for Plotly charts.
- `[app/static/app.js:fetchDashboard]` - Fetches dashboard state (`forceRefresh = false` calls `GET /api/dashboard`; `true` calls `POST /api/refresh`) and updates summary cards, gauge status, forecast table, and hydrograph.
- `[app/static/app.js:updateSummaryCards]` - Updates alert banner and top-level KPI summary cards.
- `[app/static/app.js:renderHydrograph]` - Renders the observed 72-hour streamway depth, surface rain bars, and 48-hour P10/P50/P90 + NWP ensemble fan in `#hydrograph-chart`.
- `[app/static/app.js:renderGaugeStatus]` - Renders telemetry health cards and NWP ensemble member breakdown in `#gauge-status-list`.
- `[app/static/app.js:renderForecastTable]` - Populates `#forecast-table-body` with checkpoint rows across `+1h` to `+48h`.
- `[app/static/app.js:loadVerification]` - Fetches `/api/verification` for the selected horizon and time window.
- `[app/static/app.js:renderVerification]` - Renders the historical verification hydrograph, red-event alarm classification table, flood attribution bar chart, and probability calibration scatter chart.
- `[app/static/app.js:inspectTargetForecastEvolution]` - Queries `/api/forecast-evolution` for a target UTC timestamp and renders lead-time convergence in `#drilldown-table-body`.
- `[app/static/app.js:spotlightVerificationEvent]` - Zooms `#verif-hydro-chart` around a selected verification event timestamp and opens the lead-time evolution drill-down.
- `[app/static/app.js:openDrilldownBtn]` - DOM button trigger to open the manual target timestamp drill-down inspector.
- `[app/static/app.js:drilldownGoBtn]` - DOM button trigger to execute target timestamp forecast evolution lookup.
- `[app/static/app.js:runSimulation]` - Queries `/api/simulate` using slider parameters and renders the synthetic storm response.
- `[app/static/app.js:renderSimulatorChart]` - Renders the 48-hour synthetic storm trajectory and P10–P90 envelope in `#sim-chart`.
- `[app/static/app.js:loadDiagnostics]` - Fetches `/api/diagnostics` once and triggers diagnostic chart rendering.
- `[app/static/app.js:renderDiagnosticsCharts]` - Renders CCF (`#ccf-chart`), ACF (`#acf-chart`), dry-rain scatter (`#dry-rain-chart`), and horizon NSE comparison (`#horizon-chart`).
- `[app/static/index.html:alert-banner]` - Caver safety status banner (`GREEN`, `AMBER`, `RED`).
- `[app/static/index.html:tab-forecast]`, `[app/static/index.html:tab-verification]`, `[app/static/index.html:tab-simulator]`, `[app/static/index.html:tab-diagnostics]` - Top-level navigation tabs.
- `[app/static/index.html:panel-forecast]`, `[app/static/index.html:hydrograph-chart]`, `[app/static/index.html:gauge-status-list]`, `[app/static/index.html:forecast-table-body]` - Live forecast view containers.

## Invariants
- `[app/static/index.html:color-scheme]` Declares `<meta name="color-scheme" content="light dark" />` to support both light and dark OS themes without FOUC.
- `[app/static/app.js:updateSummaryCards]` Classifies peat bog saturation `c.moisture_pct` as `Saturated (High Runoff)` when `>= 65`, `Dry Sponge` when `<= 35`, and `Moderate` otherwise.
- `[app/static/app.js:renderHydrograph]` Plots horizontal reference lines at `550` mm (The Step datum) and `700` mm (Amber/Red threshold, ~6" over The Step), and prepends the latest observed point `hist[hist.length - 1]` to forecast traces (`fcX`, `fcP50`, `fcP10`, `fcP90`, `fcP90Hydro`).
- `[app/static/app.js:renderGaugeStatus]` Iterates over fixed gauge keys `["depth_sensor", "stump_gauge", "swcc_gauge", "nwp_ensemble"]` and assigns `chip-online` iff `g.status.startsWith("ONLINE")` (else `chip-warn`).
- `[app/static/app.js:renderForecastTable]` Filters `forecast48h` rows to fixed checkpoints `[1, 2, 3, 6, 12, 18, 24, 36, 48]` hours and maps `row.status` `RED` -> `chip-danger`, `AMBER` -> `chip-warn`, otherwise `chip-online`.
- `[app/static/app.js:renderVerification]` Draws the train/test cutoff line and shaded training region on `verif-hydro-chart` when `meta.split_cutoff_iso` lies strictly between `tsX[0]` and `tsX[tsX.length - 1]`.
- `[app/static/app.js:spotlightVerificationEvent]` Zooms `verif-hydro-chart` x-axis range to `±24` hours (`24 * 3600 * 1000` ms) around `targetIso`.
- `[app/static/app.js:runSimulation]` Classifies `moist` slider value as `Winter Wet` (`>= 65`), `Summer Dry` (`<= 35`), or `Transitional`.
- `[app/static/app.js:renderSimulatorChart]` Displays `RED FLOOD PULSE` if `sim.crosses_700mm_p50` is truthy, `AMBER RISK` if `sim.crosses_700mm_p90` is truthy, otherwise `GREEN / CONTAINED`.
- `[app/static/app.js:renderDiagnosticsCharts]` Plots the `horizon-chart` reference threshold line at NSE `0.75` across 4 fixed model series (`Persistence`, `Telemetry-Only (No Future Wx)`, `NWP-Coupled (Realistic Wx Error)`, `NWP-Coupled (Exact Wx Upper Bound)`).

## Side Effects
- `[app/static/app.js:darkThemeMedia]` Registers a `change` listener on `window.matchMedia("(prefers-color-scheme: dark)")` that updates `isDark` and re-renders active Plotly charts (`renderHydrograph`, `renderVerification`, `renderSimulatorChart`, `renderDiagnosticsCharts`).
- `[app/static/app.js:fetchDashboard]` Issues `GET /api/dashboard` (or `POST /api/refresh` when `forceRefresh` is true), mutates `latestDashboardData`, and updates DOM elements and Plotly chart `#hydrograph-chart`.
- `[app/static/app.js:tab-btn]` Attaches `click` listeners to `.tab-btn` elements to toggle `aria-selected` and `.tab-panel` visibility, lazily triggering `loadVerification()`, `runSimulation()`, or `loadDiagnostics()` on first open.
- `[app/static/app.js:loadVerification]` Issues `GET /api/verification?horizon_h=${h}&window=${win}`, mutates `latestVerificationData`, and renders Plotly charts `#verif-hydro-chart`, `#verif-cause-chart`, and `#verif-prob-scatter-chart` plus verification tables.
- `[app/static/app.js:inspectTargetForecastEvolution]` Issues `GET /api/forecast-evolution?target_timestamp=${encodeURIComponent(targetIso)}` and populates `#target-drilldown-box` and `#drilldown-table-body`.
- `[app/static/app.js:runSimulation]` Issues `GET /api/simulate?rain_6h_mm=${rain}&moisture_pct=${moist}&start_depth_mm=${depth}` on slider `input` or `.sim-preset` `click`, mutates `latestSimData`, and renders Plotly chart `#sim-chart`.
- `[app/static/app.js:loadDiagnostics]` Issues `GET /api/diagnostics`, mutates `latestDiagnosticsData`, and renders Plotly charts `#ccf-chart`, `#acf-chart`, `#dry-rain-chart`, and `#horizon-chart`.
- `[app/static/app.js:DOMContentLoaded]` Attaches `DOMContentLoaded` listener on `window` to call `fetchDashboard(false)` and `click` listener on `#refresh-btn` to call `fetchDashboard(true)`.

## Verification
- Run: `pytest`
