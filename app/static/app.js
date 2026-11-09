const darkThemeMedia = window.matchMedia("(prefers-color-scheme: dark)");
let isDark = darkThemeMedia.matches;
let latestDashboardData = null;
let latestDiagnosticsData = null;
let latestSimData = null;
let latestVerificationData = null;

function getPlotlyTheme() {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: {
      color: isDark ? "#f8fafc" : "#0f172a",
      family: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    },
    hoverlabel: {
      bgcolor: isDark ? "#0f172a" : "#ffffff",
      bordercolor: isDark ? "#475569" : "#cbd5e1",
      font: {
        color: isDark ? "#f8fafc" : "#0f172a",
        family: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        size: 12,
      },
      namelength: -1,
    },
    gridcolor: isDark ? "rgba(148, 163, 184, 0.16)" : "rgba(15, 23, 42, 0.1)",
  };
}

// Listen for OS light/dark theme changes as required by modern-web-guidance
darkThemeMedia.addEventListener("change", (e) => {
  isDark = e.matches;
  if (latestDashboardData) renderHydrograph(latestDashboardData);
  if (latestVerificationData) renderVerification(latestVerificationData);
  if (latestSimData) renderSimulatorChart(latestSimData);
  if (latestDiagnosticsData) renderDiagnosticsCharts(latestDiagnosticsData);
});

async function fetchDashboard(forceRefresh = false) {
  const syncBadge = document.getElementById("sync-badge");
  syncBadge.textContent = forceRefresh ? "Refreshing ThingSpeak & NWP…" : "Loading telemetry…";
  try {
    const resp = await fetch(forceRefresh ? "/api/refresh" : "/api/dashboard", {
      method: forceRefresh ? "POST" : "GET",
    });
    const data = await resp.json();
    latestDashboardData = data;
    updateSummaryCards(data);
    renderHydrograph(data);
    renderGaugeStatus(data.gauges, data.nwp_ensemble);
    renderForecastTable(data.forecast_48h);
  } catch (err) {
    syncBadge.textContent = "Sync error — retrying";
    console.error("Failed to fetch dashboard:", err);
  }
}

function updateSummaryCards(data) {
  const c = data.current;
  document.getElementById("sync-badge").textContent =
    `Updated: ${data.generated_at_utc} (${data.sync_status})`;

  // Alert Banner
  const banner = document.getElementById("alert-banner");
  const pill = document.getElementById("alert-pill");
  banner.className = `alert-banner alert-${c.alert_level.toLowerCase()}`;
  pill.textContent = c.alert_level;
  document.getElementById("alert-headline").textContent = c.alert_headline;
  const nwpLiftNote =
    c.max_nwp_lift_48h_mm && c.max_nwp_lift_48h_mm > 0
      ? ` • 48h Crest: ${c.max_p50_48h_mm} / ${c.max_p90_48h_mm} mm (+${c.max_nwp_lift_48h_mm} mm NWP Ensemble Lift)`
      : "";
  document.getElementById("alert-subtext").textContent =
    `Latest sensor reading at ${c.timestamp.slice(0, 16).replace("T", " ")} UTC • 24h Forecast Crest: ${c.max_p50_24h_mm} mm (P50) / ${c.max_p90_24h_mm} mm (P90)${nwpLiftNote}`;
  document.getElementById("countdown-label").textContent = c.countdown_label;
  document.getElementById("countdown-value").textContent = c.countdown_value;

  // KPI 1: Depth
  document.getElementById("kpi-depth-mm").textContent = `${c.depth_mm} mm`;
  document.getElementById("kpi-regime-tag").textContent = c.hydro_regime;
  const signIn = c.step_over_inches >= 0 ? "+" : "";
  document.getElementById("kpi-step-inches").textContent =
    `${signIn}${c.step_over_inches}" (${signIn}${c.step_over_mm} mm) vs. The Step (550 mm)`;

  // KPI 2: Rate of Change
  const s1 = c.ddepth_1h_mm >= 0 ? "+" : "";
  const s3 = c.ddepth_3h_mm >= 0 ? "+" : "";
  document.getElementById("kpi-ddepth").textContent =
    `${s1}${c.ddepth_1h_mm} mm/h (${s3}${c.ddepth_3h_mm} / 3h)`;
  document.getElementById("kpi-baseflow").textContent =
    `30d 5th-Pct Baseflow Datum: ${c.baseflow_30d_mm} mm`;

  // KPI 3: Peat Bog Saturation
  document.getElementById("kpi-moisture").textContent = `${c.moisture_pct}%`;
  document.getElementById("kpi-peat-tag").textContent =
    c.moisture_pct >= 65
      ? "Saturated (High Runoff)"
      : c.moisture_pct <= 35
        ? "Dry Sponge"
        : "Moderate";
  document.getElementById("kpi-rain-recent").textContent =
    `Recent Rain: ${c.rain_6h_mm} mm (6h) • ${c.rain_24h_mm} mm (24h)`;

  // KPI 4: 24h Forecast Crest
  document.getElementById("kpi-peak-24h").textContent =
    `${c.max_p50_24h_mm} / ${c.max_p90_24h_mm} mm`;
  document.getElementById("kpi-surface-wx").textContent =
    `Fan Gyhirych: ${c.temp_c}°C • Wind ${c.wind_kmh} km/h • ${c.press_hpa} hPa`;
}

function renderHydrograph(data) {
  const theme = getPlotlyTheme();
  const hist = data.history_72h;
  const fc = data.forecast_48h;
  const nowPt = hist[hist.length - 1];

  const histX = hist.map((d) => d.timestamp);
  const histY = hist.map((d) => d.depth_mm);
  const histRain = hist.map((d) => d.fused_rain_mm);

  const fcX = [nowPt.timestamp, ...fc.map((d) => d.timestamp)];
  const fcP50 = [nowPt.depth_mm, ...fc.map((d) => d.p50_mm)];
  const fcP10 = [nowPt.depth_mm, ...fc.map((d) => d.p10_mm)];
  const fcP90 = [nowPt.depth_mm, ...fc.map((d) => d.p90_mm)];
  const fcP90Hydro = [nowPt.depth_mm, ...fc.map((d) => d.p90_hydro_only_mm ?? d.p90_mm)];
  const fcRainX = fc.map((d) => d.timestamp);
  const fcRainY = fc.map((d) => d.rain_mm);
  const fcRainP90Y = fc.map((d) => d.rain_p90_mm ?? d.rain_mm);
  const fcRainPop = fc.map((d) => d.precip_prob_pct ?? 0);

  const traces = [
    // P90 Upper Bound (Hydro + NWP Ensemble)
    {
      x: fcX,
      y: fcP90,
      mode: "lines",
      line: { width: 1.5, color: "rgba(245, 158, 11, 0.85)", dash: "dot" },
      name: "P90 Safety Bound (Hydro + NWP Ensemble)",
      hovertemplate: "%{y:.1f} mm",
      xaxis: "x",
      yaxis: "y",
    },
    // P10 Lower Bound filled to P90
    {
      x: fcX,
      y: fcP10,
      mode: "lines",
      line: { width: 0 },
      fill: "tonexty",
      fillcolor: "rgba(59, 130, 246, 0.22)",
      name: "P10 Lower Bound (80% Band)",
      hovertemplate: "%{y:.1f} mm",
      xaxis: "x",
      yaxis: "y",
    },
    // P90 Hydrological-Only Reference (Deterministic NWP)
    {
      x: fcX,
      y: fcP90Hydro,
      mode: "lines",
      line: { width: 1.2, color: "rgba(148, 163, 184, 0.7)", dash: "dash" },
      name: "P90 (Deterministic NWP Only)",
      hovertemplate: "%{y:.1f} mm",
      xaxis: "x",
      yaxis: "y",
    },
    // Observed Past 72h Depth
    {
      x: histX,
      y: histY,
      mode: "lines",
      name: "Observed Depth",
      line: { color: isDark ? "#38bdf8" : "#0f172a", width: 2.5 },
      hovertemplate: "%{y:.1f} mm",
      xaxis: "x",
      yaxis: "y",
    },
    // Forecast P50 Median
    {
      x: fcX,
      y: fcP50,
      mode: "lines",
      name: "Forecast P50 Median",
      line: { color: "#10b981", width: 2.6 },
      hovertemplate: "%{y:.1f} mm",
      xaxis: "x",
      yaxis: "y",
    },
    // Observed Surface Rain (Bottom Subplot)
    {
      x: histX,
      y: histRain,
      type: "bar",
      name: "Observed Rain",
      marker: { color: "#3b82f6" },
      hovertemplate: "%{y:.2f} mm/h",
      xaxis: "x",
      yaxis: "y2",
    },
    // Forecast NWP Rain P50 Median (Bottom Subplot)
    {
      x: fcRainX,
      y: fcRainY,
      type: "bar",
      name: "NWP Rain (P50 Median)",
      marker: { color: "#a855f7" },
      hovertemplate: "%{y:.2f} mm/h",
      xaxis: "x",
      yaxis: "y2",
    },
    // Forecast NWP Ensemble P90 Upper Bound (Bottom Subplot)
    {
      x: fcRainX,
      y: fcRainP90Y,
      customdata: fcRainPop,
      mode: "lines",
      name: "NWP Ensemble P90 Rain",
      line: { color: "#f59e0b", width: 1.8, dash: "dot" },
      hovertemplate: "%{y:.2f} mm/h (PoP: %{customdata}%)",
      xaxis: "x",
      yaxis: "y2",
    },
  ];

  const layout = {
    paper_bgcolor: theme.paper_bgcolor,
    plot_bgcolor: theme.plot_bgcolor,
    font: theme.font,
    hoverlabel: theme.hoverlabel,
    margin: { t: 24, r: 24, b: 48, l: 58 },
    hovermode: "x unified",
    legend: { orientation: "h", y: 1.08, x: 0 },
    xaxis: {
      anchor: "y2",
      gridcolor: theme.gridcolor,
      title: "Time (UTC)",
    },
    yaxis: {
      domain: [0.30, 1.0],
      title: "Streamway Depth (mm)",
      gridcolor: theme.gridcolor,
    },
    yaxis2: {
      domain: [0.0, 0.22],
      title: "Rain (mm/h)",
      gridcolor: theme.gridcolor,
    },
    shapes: [
      // 700 mm Amber/Red Threshold
      {
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        yref: "y",
        y0: 700,
        y1: 700,
        line: { color: "#ef4444", width: 2, dash: "dash" },
      },
      // 550 mm Step Datum
      {
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        yref: "y",
        y0: 550,
        y1: 550,
        line: { color: "#64748b", width: 1, dash: "dot" },
      },
      // Vertical "NOW" divider
      {
        type: "line",
        xref: "x",
        x0: nowPt.timestamp,
        x1: nowPt.timestamp,
        yref: "paper",
        y0: 0,
        y1: 1,
        line: { color: "#f59e0b", width: 1.5, dash: "dashdot" },
      },
    ],
    annotations: [
      {
        xref: "paper",
        x: 0.01,
        yref: "y",
        y: 715,
        text: '700 mm Amber/Red Threshold (~6" over The Step)',
        showarrow: false,
        font: { color: "#ef4444", size: 11 },
      },
    ],
  };

  Plotly.react("hydrograph-chart", traces, layout, { responsive: true, displayModeBar: false });
}

function renderGaugeStatus(gauges, nwpEnsemble) {
  const container = document.getElementById("gauge-status-list");
  container.innerHTML = "";

  for (const key of ["depth_sensor", "stump_gauge", "swcc_gauge", "nwp_ensemble"]) {
    const g = gauges[key];
    if (!g) continue;
    const isOnline = g.status.startsWith("ONLINE");
    const chipClass = isOnline ? "chip-online" : "chip-warn";
    const extra = g.glitch_filter ? ` • ${g.glitch_filter}` : "";

    let memberHtml = "";
    if (key === "nwp_ensemble" && nwpEnsemble && Array.isArray(nwpEnsemble.members) && nwpEnsemble.members.length > 0) {
      const pills = nwpEnsemble.members
        .map((m) => `<span class="kpi-tag" style="display:inline-block;margin:2px 4px 0 0">${m.label}: ${m.rain_48h_mm} mm (48h)</span>`)
        .join("");
      memberHtml = `<div style="margin-top:0.35rem">${pills}</div>`;
    }

    const item = document.createElement("div");
    item.className = "gauge-item";
    item.innerHTML = `
      <div>
        <div class="gauge-name">${g.name}</div>
        <div class="gauge-meta">Cadence: ${g.cadence} • ${g.last_seen}${extra}</div>
        ${memberHtml}
      </div>
      <span class="status-chip ${chipClass}">${g.status}</span>
    `;
    container.appendChild(item);
  }
}

function renderForecastTable(forecast48h) {
  const checkpoints = [1, 2, 3, 6, 12, 18, 24, 36, 48];
  const tbody = document.getElementById("forecast-table-body");
  tbody.innerHTML = "";

  for (const row of forecast48h) {
    if (!checkpoints.includes(row.horizon_h)) continue;
    const chipClass =
      row.status === "RED"
        ? "chip-danger"
        : row.status === "AMBER"
          ? "chip-warn"
          : "chip-online";
    const signIn = row.step_inches_p50 >= 0 ? "+" : "";
    const nwpLiftBadge =
      row.nwp_uncertainty_mm && row.nwp_uncertainty_mm >= 1.0
        ? ` <small style="color:var(--text-secondary)">(+${row.nwp_uncertainty_mm} mm NWP)</small>`
        : "";
    const rP50 = (row.rain_mm ?? 0).toFixed(2);
    const rP90 = (row.rain_p90_mm ?? row.rain_mm ?? 0).toFixed(2);
    const pop = row.precip_prob_pct ?? 0;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <th scope="row">+${row.horizon_h}h</th>
      <td>${row.timestamp.slice(5, 16).replace("T", " ")}</td>
      <td><strong>${row.p50_mm} mm</strong></td>
      <td>${row.p90_mm} mm${nwpLiftBadge}</td>
      <td>${rP50} / ${rP90} mm (${pop}%)</td>
      <td>${signIn}${row.step_inches_p50}&quot;</td>
      <td><span class="status-chip ${chipClass}">${row.status}</span></td>
    `;
    tbody.appendChild(tr);
  }
}

// Tab switching
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    document.querySelectorAll(".tab-panel").forEach((p) => {
      p.hidden = true;
    });

    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    const panel = document.getElementById(btn.dataset.target);
    panel.hidden = false;

    if (btn.dataset.target === "panel-verification" && !latestVerificationData) {
      loadVerification();
    } else if (btn.dataset.target === "panel-simulator" && !latestSimData) {
      runSimulation();
    } else if (btn.dataset.target === "panel-diagnostics" && !latestDiagnosticsData) {
      loadDiagnostics();
    }
  });
});

// Historical Verification Ledger logic
async function loadVerification() {
  const h = document.getElementById("verif-horizon").value;
  const win = document.getElementById("verif-window").value;
  const resp = await fetch(`/api/verification?horizon_h=${h}&window=${win}`);
  const verif = await resp.json();
  latestVerificationData = verif;
  renderVerification(verif);
}

function renderVerification(verif) {
  const m = verif.selected_metrics || {};
  const meta = verif.data_split_metadata || {};
  const theme = getPlotlyTheme();

  // Populate Training vs Out-of-Sample Provenance Banner
  if (meta.train_split && meta.test_split) {
    const tr = meta.train_split;
    const te = meta.test_split;
    document.getElementById("split-train-dates").textContent =
      `${tr.start_utc} → ${tr.end_utc} (${tr.hours.toLocaleString()} hrs)`;
    document.getElementById("split-train-stats").textContent =
      `+${verif.selected_horizon_h}h In-Sample Fit: MAE = ${tr.mae_mm} mm • RMSE = ${tr.rmse_mm} mm • NSE = ${tr.nse} • Band Cov = ${tr.cov_pct}%`;

    document.getElementById("split-test-dates").textContent =
      `${te.start_utc} → ${te.end_utc} (${te.hours.toLocaleString()} hrs)`;
    document.getElementById("split-test-stats").textContent =
      `+${verif.selected_horizon_h}h Out-of-Sample Test: MAE = ${te.mae_mm} mm • RMSE = ${te.rmse_mm} mm • NSE = ${te.nse} • Band Cov = ${te.cov_pct}%`;
  }

  // Populate Sensor Coverage Provenance Table
  const sensorBody = document.getElementById("sensor-spans-body");
  if (sensorBody && meta.sensor_spans) {
    sensorBody.innerHTML = "";
    for (const sp of meta.sensor_spans) {
      const trEl = document.createElement("tr");
      trEl.innerHTML = `
        <th scope="row">${sp.stream}</th>
        <td>${sp.start_utc}</td>
        <td>${sp.end_utc}</td>
        <td><strong>${sp.hours.toLocaleString()} hrs</strong></td>
        <td>${sp.role}</td>
      `;
      sensorBody.appendChild(trEl);
    }
  }

  document.getElementById("verif-kpi-horizon-tag").textContent = `+${verif.selected_horizon_h}h Lead`;
  document.getElementById("verif-kpi-mae").textContent =
    `${m.mae_mm ?? "—"} mm (${m.mae_inches ?? "—"}")`;
  document.getElementById("verif-kpi-rmse").textContent = `RMSE: ${m.rmse_mm ?? "—"} mm`;
  document.getElementById("verif-kpi-nse").textContent = `${m.nse ?? "—"}`;
  document.getElementById("verif-kpi-skill").textContent =
    `Skill vs. Naive Persistence: +${m.skill_vs_pers_pct ?? "—"}%`;
  document.getElementById("verif-kpi-count-tag").textContent =
    `${m.verified_hours ?? 0} verified hrs`;
  document.getElementById("verif-kpi-cov").textContent = `${m.band_coverage_pct ?? "—"}%`;
  document.getElementById("verif-kpi-recall").textContent =
    `${m.recall_p50_pct ?? "—"}% (P50) / ${m.recall_p90_pct ?? "—"}% (P90)`;
  document.getElementById("verif-kpi-hits").textContent =
    `≥700mm Hits: ${m.hits_p50 ?? 0} • False Alarms: ${m.false_alarms_p50 ?? 0} • Missed: ${m.missed_p50 ?? 0}`;

  // Plot Predicted (P50 + P10-P90 band) vs Actual Measured Depth + Error vs P10-P90 Envelope + Historical NWP PoP
  const s = verif.series || [];
  const pa = verif.probability_attribution || {};
  const tsX = s.map((d) => d.target_timestamp);
  const actY = s.map((d) => d.actual_depth_mm);
  const p50Y = s.map((d) => d.pred_p50_mm);
  const p10Y = s.map((d) => d.pred_p10_mm);
  const p90Y = s.map((d) => d.pred_p90_mm);
  const persY = s.map((d) => d.start_depth_mm);
  const errY = s.map((d) => d.error_mm);
  const errLowY = s.map((d) => d.err_low_bound_mm ?? d.pred_p10_mm - d.pred_p50_mm);
  const errHighY = s.map((d) => d.err_high_bound_mm ?? d.pred_p90_mm - d.pred_p50_mm);
  const errBarColors = s.map((d) =>
    d.in_band ? "rgba(16, 185, 129, 0.78)" : "rgba(249, 115, 22, 0.88)"
  );
  const popY = s.map((d) => (d.hist_pop_6h_pct !== null && d.hist_pop_6h_pct !== undefined ? d.hist_pop_6h_pct : null));
  const nwpSpCustom = s.map((d) => [
    d.hist_nwp_spread_6h_mm !== null && d.hist_nwp_spread_6h_mm !== undefined
      ? `Δ${d.hist_nwp_spread_6h_mm} mm`
      : "n/a",
    d.club_rain_6h_mm ?? 0,
    d.grid_rain_6h_mm ?? 0,
  ]);

  const topDevs = pa.top_deviations || [];
  const devMarkerX = topDevs.map((d) => d.target_iso || d.target_timestamp.replace(" ", "T") + ":00+00:00");
  const devMarkerY = topDevs.map((d) => d.actual_depth_mm);
  const devMarkerColors = topDevs.map((d) => d.color || (d.in_band ? "#10b981" : "#f59e0b"));
  const devMarkerCustom = topDevs.map((d) => [
    d.pred_p50_mm,
    d.error_mm >= 0 ? `+${d.error_mm}` : `${d.error_mm}`,
    d.pred_band,
    d.hist_pop_6h_pct !== null && d.hist_pop_6h_pct !== undefined ? `${d.hist_pop_6h_pct}%` : "n/a",
    d.hist_nwp_spread_6h_mm !== null && d.hist_nwp_spread_6h_mm !== undefined
      ? `Δ${d.hist_nwp_spread_6h_mm} mm`
      : "n/a",
    d.diagnosis_tag,
    d.diagnosis_detail,
  ]);

  const traces = [
    {
      x: tsX,
      y: p90Y,
      mode: "lines",
      line: { width: 1, color: "rgba(245, 158, 11, 0.45)", dash: "dot" },
      name: `Predicted P90 (+${verif.selected_horizon_h}h)`,
      hovertemplate: "%{y:.1f} mm",
      xaxis: "x",
      yaxis: "y",
    },
    {
      x: tsX,
      y: p10Y,
      mode: "lines",
      line: { width: 0 },
      fill: "tonexty",
      fillcolor: "rgba(59, 130, 246, 0.22)",
      name: `Predicted P10 (+${verif.selected_horizon_h}h)`,
      hovertemplate: "%{y:.1f} mm",
      xaxis: "x",
      yaxis: "y",
    },
    {
      x: tsX,
      y: persY,
      mode: "lines",
      name: `Naive Persistence (+${verif.selected_horizon_h}h)`,
      line: { color: "#94a3b8", width: 1.3, dash: "dot" },
      hovertemplate: "%{y:.1f} mm",
      xaxis: "x",
      yaxis: "y",
    },
    {
      x: tsX,
      y: actY,
      mode: "lines",
      name: "Actual Measured Depth",
      line: { color: isDark ? "#f8fafc" : "#0f172a", width: 2.3 },
      hovertemplate: "%{y:.1f} mm",
      xaxis: "x",
      yaxis: "y",
    },
    {
      x: tsX,
      y: p50Y,
      mode: "lines",
      name: `Predicted P50 (+${verif.selected_horizon_h}h)`,
      line: { color: "#3b82f6", width: 2.1 },
      hovertemplate: "%{y:.1f} mm",
      xaxis: "x",
      yaxis: "y",
    },
    {
      x: devMarkerX,
      y: devMarkerY,
      customdata: devMarkerCustom,
      mode: "markers",
      name: "Top Deviations (Explained)",
      marker: {
        size: 10,
        color: devMarkerColors,
        symbol: "diamond",
        line: { width: 1.5, color: isDark ? "#0f172a" : "#ffffff" },
      },
      hovertemplate:
        "<b>%{customdata[5]}</b><br>Actual: %{y:.1f} mm vs P50: %{customdata[0]} mm (%{customdata[1]} mm)<br>P10–P90 Band: %{customdata[2]} • Archive PoP: %{customdata[3]} (NWP %{customdata[4]})<br><i>%{customdata[6]}</i><extra></extra>",
      xaxis: "x",
      yaxis: "y",
    },
    // Subplot 2: P10-P90 Error Envelope + Signed Error Bars
    {
      x: tsX,
      y: errHighY,
      mode: "lines",
      line: { width: 1, color: "rgba(16, 185, 129, 0.35)", dash: "dot" },
      name: "P10–P90 Error Tolerance Envelope",
      hovertemplate: "Upper Band Tolerance: %{y:+.1f} mm<extra></extra>",
      showlegend: true,
      xaxis: "x",
      yaxis: "y2",
    },
    {
      x: tsX,
      y: errLowY,
      mode: "lines",
      line: { width: 1, color: "rgba(16, 185, 129, 0.35)", dash: "dot" },
      fill: "tonexty",
      fillcolor: "rgba(16, 185, 129, 0.16)",
      name: "P10–P90 Lower Tolerance",
      hovertemplate: "Lower Band Tolerance: %{y:+.1f} mm<extra></extra>",
      showlegend: false,
      xaxis: "x",
      yaxis: "y2",
    },
    {
      x: tsX,
      y: errY,
      type: "bar",
      name: "Error (Green = In P10–P90 Band, Orange = Outside)",
      marker: { color: errBarColors },
      hovertemplate: "Error (P50−Actual): %{y:+.1f} mm<extra></extra>",
      xaxis: "x",
      yaxis: "y2",
    },
    // Subplot 3: Historical 6h NWP Precipitation Probability & Model Spread
    {
      x: tsX,
      y: popY,
      customdata: nwpSpCustom,
      type: "scatter",
      mode: "lines",
      fill: "tozeroy",
      fillcolor: "rgba(14, 165, 233, 0.20)",
      line: { color: "#0ea5e9", width: 1.6 },
      name: "Archive 6h PoP (%) & Issue-Lead NWP Spread",
      hovertemplate:
        "Archive 6h Max PoP: <b>%{y:.0f}%</b> • Issue-Lead NWP Spread: <b>%{customdata[0]}</b> (Gauge %{customdata[1]:.1f} / Grid %{customdata[2]:.1f} mm)<extra></extra>",
      xaxis: "x",
      yaxis: "y3",
    },
  ];

  const shapes = [
    {
      type: "line",
      xref: "paper",
      x0: 0,
      x1: 1,
      yref: "y",
      y0: 700,
      y1: 700,
      line: { color: "#ef4444", width: 2, dash: "dash" },
    },
  ];
  const annotations = [];

  // If the selected window spans the 2026-01-01 Train/Test boundary, draw the cutoff line & shaded training box
  if (meta.split_cutoff_iso && tsX.length > 1) {
    const firstTs = tsX[0];
    const lastTs = tsX[tsX.length - 1];
    if (firstTs < meta.split_cutoff_iso && lastTs > meta.split_cutoff_iso) {
      shapes.push(
        {
          type: "rect",
          xref: "x",
          x0: firstTs,
          x1: meta.split_cutoff_iso,
          yref: "paper",
          y0: 0,
          y1: 1,
          fillcolor: "rgba(245, 158, 11, 0.08)",
          line: { width: 0 },
        },
        {
          type: "line",
          xref: "x",
          x0: meta.split_cutoff_iso,
          x1: meta.split_cutoff_iso,
          yref: "paper",
          y0: 0,
          y1: 1,
          line: { color: "#f59e0b", width: 2.5, dash: "dashdot" },
        }
      );
      annotations.push({
        xref: "x",
        x: meta.split_cutoff_iso,
        yref: "paper",
        y: 0.98,
        text: "◀ Training Data (2024–2025) | Out-of-Sample Test (2026) ▶",
        showarrow: false,
        font: { color: "#f59e0b", size: 12 },
        bgcolor: isDark ? "rgba(15,23,42,0.85)" : "rgba(255,255,255,0.85)",
      });
    }
  }

  const layout = {
    paper_bgcolor: theme.paper_bgcolor,
    plot_bgcolor: theme.plot_bgcolor,
    font: theme.font,
    hoverlabel: theme.hoverlabel,
    margin: { t: 28, r: 24, b: 48, l: 58 },
    hovermode: "x unified",
    legend: { orientation: "h", y: 1.09, x: 0 },
    xaxis: {
      anchor: "y3",
      title: "Target Timestamp (UTC)",
      gridcolor: theme.gridcolor,
    },
    yaxis: {
      domain: [0.44, 1.0],
      title: "Depth (mm)",
      gridcolor: theme.gridcolor,
    },
    yaxis2: {
      domain: [0.19, 0.39],
      title: "Error vs Band",
      gridcolor: theme.gridcolor,
    },
    yaxis3: {
      domain: [0.0, 0.14],
      title: "Hist PoP %",
      range: [0, 105],
      gridcolor: theme.gridcolor,
    },
    shapes,
    annotations,
  };

  Plotly.react("verif-hydro-chart", traces, layout, { responsive: true, displayModeBar: false });

  const resetZoomBtn = document.getElementById("verif-reset-zoom-btn");
  if (resetZoomBtn) {
    resetZoomBtn.style.display = "none";
    resetZoomBtn.onclick = () => {
      Plotly.relayout("verif-hydro-chart", {
        "xaxis.autorange": true,
      });
      resetZoomBtn.style.display = "none";
    };
  }

  function spotlightVerificationEvent(targetIso) {
    if (!targetIso) return;
    const tMs = new Date(targetIso).getTime();
    if (Number.isNaN(tMs)) return;
    const startIso = new Date(tMs - 24 * 3600 * 1000).toISOString();
    const endIso = new Date(tMs + 24 * 3600 * 1000).toISOString();
    Plotly.relayout("verif-hydro-chart", {
      "xaxis.range": [startIso, endIso],
    });
    if (resetZoomBtn) resetZoomBtn.style.display = "inline-flex";
    const chartEl = document.getElementById("verif-hydro-chart");
    if (chartEl) chartEl.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  // Multi-Horizon Scorecard Table
  const scBody = document.getElementById("verif-scorecard-body");
  scBody.innerHTML = "";
  for (const row of verif.horizon_scorecard || []) {
    const isSel = row.horizon_h === verif.selected_horizon_h;
    const tr = document.createElement("tr");
    if (isSel) tr.style.backgroundColor = "rgba(59, 130, 246, 0.10)";
    tr.innerHTML = `
      <th scope="row">+${row.horizon_h}h</th>
      <td><strong>${row.mae_mm} mm</strong> (${row.mae_inches}&quot;)</td>
      <td>${row.rmse_mm} mm</td>
      <td>${row.nse}</td>
      <td>+${row.skill_vs_pers_pct}%</td>
      <td>${row.recall_p90_pct}% (P50: ${row.recall_p50_pct}%)</td>
    `;
    scBody.appendChild(tr);
  }

  // Probability & Root-Cause Attribution Panel
  const attrDevsTag = document.getElementById("attr-devs-tag");
  if (attrDevsTag) {
    attrDevsTag.textContent = `${pa.deviations_over_20mm ?? 0} hrs ≥20mm error`;
    document.getElementById("attr-in-band-pct").textContent =
      `${pa.captured_by_band_pct ?? 100}% (${pa.captured_by_band_count ?? 0}/${pa.deviations_over_20mm ?? 0})`;
    document.getElementById("attr-band-width-note").textContent =
      `Avg P10–P90 width widened from ${pa.mean_band_width_calm_mm ?? 0} mm (calm) → ${pa.mean_band_width_deviations_mm ?? 0} mm (storms)`;
    document.getElementById("attr-1h-pct").textContent =
      `${pa.captured_with_1h_lag_pct ?? 100}% (${pa.captured_with_1h_lag_count ?? 0}/${pa.deviations_over_20mm ?? 0})`;
    document.getElementById("attr-outside-pct").textContent =
      `${pa.unexplained_outside_1h_pct ?? 0}% (${pa.unexplained_outside_1h_count ?? 0}/${pa.deviations_over_20mm ?? 0})`;
    const histPopEl = document.getElementById("attr-hist-pop");
    const bt = pa.ensemble_backtest || {};
    if (histPopEl) {
      const btTag = document.getElementById("attr-bt-horizon-tag");
      if (bt.eval_horizon_h) {
        if (btTag) btTag.textContent = `+${bt.eval_horizon_h}h Backtest`;
        histPopEl.textContent = `${bt.ens_coverage_pct}% (vs ${bt.det_coverage_pct}% Det)`;
        document.getElementById("attr-hist-spread-note").textContent =
          `P50 MAE: ${bt.det_mae_mm} → ${bt.ens_mae_mm} mm • ≥700mm P90 Recall: ${bt.ens_p90_pod_pct}% (${bt.ens_p90_false_alarms} vs ${bt.det_p90_false_alarms} FA)`;
      } else {
        histPopEl.textContent =
          pa.mean_nwp_spread_deviations_mm !== null && pa.mean_nwp_spread_deviations_mm !== undefined
            ? `Δ${pa.mean_nwp_spread_deviations_mm} mm spread`
            : "n/a (pre-2024)";
        document.getElementById("attr-hist-spread-note").textContent =
          pa.mean_pop_deviations_pct !== null && pa.mean_pop_deviations_pct !== undefined
            ? `Archive 6h PoP: ${pa.mean_pop_deviations_pct}% (storms) vs. ${pa.mean_pop_calm_pct ?? 0}% (calm)`
            : "Multi-model NWP archive begins Jan 2024";
      }
    }

    // Visual 1: Contributing-Factor Breakdown Horizontal Bar Chart
    const causeEl = document.getElementById("verif-cause-chart");
    if (causeEl && Array.isArray(pa.cause_breakdown)) {
      const cb = [...pa.cause_breakdown].reverse();
      const causeTraces = [
        {
          type: "bar",
          orientation: "h",
          y: cb.map((c) => c.category),
          x: cb.map((c) => c.pct),
          marker: { color: cb.map((c) => c.color) },
          text: cb.map((c) => {
            const spTxt =
              c.mean_nwp_spread_mm !== null && c.mean_nwp_spread_mm !== undefined
                ? ` • NWP Δ${c.mean_nwp_spread_mm}mm`
                : "";
            return `${c.pct}% (${c.count}h)${spTxt}`;
          }),
          textposition: "auto",
          customdata: cb.map((c) => [
            c.count,
            c.mean_abs_err_mm,
            c.mean_band_width_mm,
            c.mean_pop_pct !== null && c.mean_pop_pct !== undefined ? `${c.mean_pop_pct}%` : "n/a",
            c.mean_nwp_spread_mm !== null && c.mean_nwp_spread_mm !== undefined
              ? `Δ${c.mean_nwp_spread_mm} mm`
              : "n/a",
          ]),
          hovertemplate:
            "<b>%{y}</b><br>Share of Deviations: <b>%{x:.1f}%</b> (%{customdata[0]} hrs)<br>Mean |Error|: %{customdata[1]} mm • Mean P10–P90 Width: %{customdata[2]} mm<br>Archive 6h PoP: <b>%{customdata[3]}</b> • Issue-Lead NWP Spread: <b>%{customdata[4]}</b><extra></extra>",
        },
      ];
      const causeLayout = {
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: theme.font,
        hoverlabel: theme.hoverlabel,
        margin: { t: 12, r: 24, b: 42, l: 215 },
        xaxis: {
          title: "Share of Deviations ≥20 mm (%)",
          range: [0, Math.max(100, ...cb.map((c) => c.pct * 1.18))],
          gridcolor: theme.gridcolor,
        },
        yaxis: { automargin: true },
      };
      Plotly.react("verif-cause-chart", causeTraces, causeLayout, {
        responsive: true,
        displayModeBar: false,
      });
    }

    // Visual 2: Probability Capture Cone vs Realized Error Scatter/Bubble Plot
    const scatterEl = document.getElementById("verif-prob-scatter-chart");
    if (scatterEl && Array.isArray(pa.deviation_scatter)) {
      const pts = pa.deviation_scatter;
      const maxBw = Math.max(200, ...pts.map((p) => p.band_width_mm || 0)) * 1.05;
      const maxErr = Math.max(120, ...pts.map((p) => p.abs_err_mm || 0)) * 1.08;

      // Cone trace: symmetric band half-width (|Error| <= 0.5 * BandWidth)
      const coneTraces = [
        {
          x: [0, maxBw],
          y: [0, maxBw * 0.55],
          mode: "lines",
          fill: "tozeroy",
          fillcolor: "rgba(16, 185, 129, 0.13)",
          line: { color: "rgba(16, 185, 129, 0.65)", width: 1.5, dash: "dash" },
          name: "P10–P90 Capture Cone",
          hoverinfo: "skip",
        },
      ];

      const categories = [
        { name: "Within P10–P90 Band (Wide Uncertainty)", color: "#10b981" },
        { name: "Flood-Wall Timing (±1h Phase Shift)", color: "#f59e0b" },
        { name: "Local Gauge > Catchment Rain", color: "#f97316" },
        { name: "Upland Orographic / Snowmelt", color: "#8b5cf6" },
        { name: "NWP Rain / Runoff Timing Error", color: "#ef4444" },
      ];

      for (const cat of categories) {
        const sub = pts.filter((p) => p.category === cat.name);
        if (!sub.length) continue;
        coneTraces.push({
          x: sub.map((p) => p.band_width_mm),
          y: sub.map((p) => p.abs_err_mm),
          mode: "markers",
          name: cat.name,
          marker: {
            color: cat.color,
            size: sub.map((p) => Math.min(22, Math.max(8, 7 + (p.hist_nwp_spread_6h_mm || 0) * 1.3))),
            opacity: 0.85,
            line: { width: 1, color: isDark ? "#0f172a" : "#ffffff" },
          },
          customdata: sub.map((p) => [
            p.target_timestamp,
            p.pred_p50_mm,
            p.actual_depth_mm,
            p.error_mm >= 0 ? `+${p.error_mm}` : `${p.error_mm}`,
            p.hist_pop_6h_pct !== null && p.hist_pop_6h_pct !== undefined ? `${p.hist_pop_6h_pct}%` : "n/a",
            p.hist_nwp_spread_6h_mm !== null && p.hist_nwp_spread_6h_mm !== undefined
              ? `Δ${p.hist_nwp_spread_6h_mm} mm`
              : "n/a",
            p.club_rain_6h_mm ?? 0,
            p.grid_rain_6h_mm ?? 0,
            p.diagnosis_tag,
            p.target_iso,
          ]),
          hovertemplate:
            "<b>%{customdata[0]} UTC</b> — %{customdata[8]}<br>|Error|: <b>%{y:.1f} mm</b> (P50 %{customdata[1]} vs Actual %{customdata[2]} mm, %{customdata[3]} mm)<br>P10–P90 Width: <b>%{x:.0f} mm</b> • Archive PoP: <b>%{customdata[4]}</b> • Issue-Lead NWP Spread: <b>%{customdata[5]}</b><br>6h Rain: Gauge %{customdata[6]} mm / Grid %{customdata[7]} mm<extra></extra>",
        });
      }

      const scatterLayout = {
        paper_bgcolor: theme.paper_bgcolor,
        plot_bgcolor: theme.plot_bgcolor,
        font: theme.font,
        hoverlabel: theme.hoverlabel,
        margin: { t: 28, r: 18, b: 42, l: 55 },
        legend: { orientation: "h", y: 1.15, x: 0, font: { size: 10.5 } },
        xaxis: {
          title: "Predicted P10–P90 Uncertainty Band Width (mm)",
          range: [0, maxBw],
          gridcolor: theme.gridcolor,
        },
        yaxis: {
          title: "|Forecast Error| (mm)",
          range: [0, maxErr],
          gridcolor: theme.gridcolor,
        },
      };

      Plotly.react("verif-prob-scatter-chart", coneTraces, scatterLayout, {
        responsive: true,
        displayModeBar: false,
      });

      if (typeof scatterEl.removeAllListeners === "function") {
        scatterEl.removeAllListeners("plotly_click");
      }
      if (typeof scatterEl.on === "function") {
        scatterEl.on("plotly_click", (ev) => {
          const cd = ev?.points?.[0]?.customdata;
          if (cd && cd[9]) {
            spotlightVerificationEvent(cd[9]);
          }
        });
      }
    }

    const devBody = document.getElementById("verif-deviations-body");
    devBody.innerHTML = "";
    for (const d of pa.top_deviations || []) {
      const signE = d.error_mm >= 0 ? "+" : "";
      const qClass = d.in_band ? "chip-online" : d.in_band_1h ? "chip-warn" : "chip-danger";
      const qLabel = d.in_band
        ? `P${d.realized_quantile_pct} (IN BAND)`
        : d.in_band_1h
          ? `P${d.realized_quantile_pct} (±1h BAND)`
          : `P${d.realized_quantile_pct} (OUTSIDE)`;
      const hasPop = d.hist_pop_6h_pct !== null && d.hist_pop_6h_pct !== undefined;
      const hasSp = d.hist_nwp_spread_6h_mm !== null && d.hist_nwp_spread_6h_mm !== undefined;
      const nwpCell =
        hasPop || hasSp
          ? `<strong>${hasPop ? `${d.hist_pop_6h_pct}% PoP` : "PoP n/a"}</strong> • ${
              hasSp
                ? `${d.hist_nwp_p10_6h_mm}–${d.hist_nwp_p90_6h_mm} mm <small style="color:var(--text-secondary)">(Δ${d.hist_nwp_spread_6h_mm} mm)</small>`
                : "n/a"
            }`
          : `<small style="color:var(--text-secondary)">n/a (pre-2024)</small>`;
      const tr = document.createElement("tr");
      tr.style.cursor = "pointer";
      tr.title = "Click to zoom hydrograph to this 48h flood event";
      tr.addEventListener("click", () => {
        spotlightVerificationEvent(d.target_iso);
      });
      tr.innerHTML = `
        <th scope="row">${d.target_timestamp} 🔍</th>
        <td><strong>${d.pred_p50_mm}</strong> vs <strong>${d.actual_depth_mm} mm</strong> (${signE}${d.error_mm} mm)</td>
        <td>${d.pred_band} <small style="color:var(--text-secondary)">(±${Math.round(d.band_width_mm / 2)} mm)</small></td>
        <td><span class="status-chip ${qClass}">${qLabel}</span></td>
        <td>${nwpCell}</td>
        <td>${d.club_rain_6h_mm} / ${d.grid_rain_6h_mm} mm</td>
        <td><strong>${d.diagnosis_tag}:</strong> ${d.diagnosis_detail}</td>
      `;
      devBody.appendChild(tr);
    }
  }

  // Recent Audit Log Table
  const auBody = document.getElementById("verif-audit-body");
  auBody.innerHTML = "";
  for (const r of verif.recent_audit_rows || []) {
    const signErr = r.error_mm >= 0 ? "+" : "";
    const chipClass = r.in_band ? "chip-online" : "chip-warn";
    const qStr = r.realized_quantile_pct !== undefined ? `P${r.realized_quantile_pct} • ` : "";
    const chipText = r.in_band ? `${qStr}IN BAND` : `${qStr}OUTSIDE`;
    const isTrain = r.source === "train_in_sample";
    const srcBadge = isTrain
      ? `<span class="status-chip chip-warn">TRAIN (24–25)</span>`
      : r.source === "live_forecast"
        ? `<span class="status-chip chip-online">LIVE</span>`
        : `<span class="status-chip chip-online">TEST (2026)</span>`;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <th scope="row">${r.target_timestamp.slice(5)}</th>
      <td>${srcBadge}</td>
      <td><strong>${r.pred_p50_mm} mm</strong></td>
      <td>${r.pred_band}</td>
      <td><strong>${r.actual_depth_mm} mm</strong></td>
      <td>${signErr}${r.error_mm} mm</td>
      <td><span class="status-chip ${chipClass}">${chipText}</span></td>
      <td><small>${r.diagnosis_tag || "—"}</small></td>
    `;
    auBody.appendChild(tr);
  }
}

["verif-horizon", "verif-window"].forEach((id) => {
  document.getElementById(id).addEventListener("change", loadVerification);
});

// Simulator logic
async function runSimulation() {
  const rain = parseFloat(document.getElementById("sim-rain").value);
  const moist = parseFloat(document.getElementById("sim-moisture").value);
  const depth = parseFloat(document.getElementById("sim-start-depth").value);

  document.getElementById("sim-rain-val").textContent = `${rain.toFixed(1)} mm`;
  const moistRegime =
    moist >= 65 ? "Winter Wet" : moist <= 35 ? "Summer Dry" : "Transitional";
  document.getElementById("sim-moisture-val").textContent = `${moist}% (${moistRegime})`;
  document.getElementById("sim-start-val").textContent = `${depth} mm`;

  const resp = await fetch(
    `/api/simulate?rain_6h_mm=${rain}&moisture_pct=${moist}&start_depth_mm=${depth}`
  );
  const sim = await resp.json();
  latestSimData = sim;
  renderSimulatorChart(sim);
}

function renderSimulatorChart(sim) {
  const theme = getPlotlyTheme();
  const statusText = sim.crosses_700mm_p50
    ? `<strong style="color:#ef4444">RED FLOOD PULSE:</strong> P50 crests at <strong>${sim.peak_p50_mm} mm</strong> (+${sim.net_rise_p50_mm} mm rise, P90 = ${sim.peak_p90_mm} mm)`
    : sim.crosses_700mm_p90
      ? `<strong style="color:#f59e0b">AMBER RISK:</strong> P50 crests at <strong>${sim.peak_p50_mm} mm</strong> (+${sim.net_rise_p50_mm} mm rise), P90 reaches <strong>${sim.peak_p90_mm} mm</strong>`
      : `<strong style="color:#10b981">GREEN / CONTAINED:</strong> Peat absorbs runoff — P50 crests at <strong>${sim.peak_p50_mm} mm</strong> (+${sim.net_rise_p50_mm} mm rise)`;

  document.getElementById("sim-summary-bar").innerHTML = statusText;

  const hrs = sim.trajectory_24h.map((d) => `+${d.hour}h`);
  const p50 = sim.trajectory_24h.map((d) => d.p50_mm);
  const p10 = sim.trajectory_24h.map((d) => d.p10_mm);
  const p90 = sim.trajectory_24h.map((d) => d.p90_mm);

  const traces = [
    {
      x: hrs,
      y: p90,
      mode: "lines",
      line: { width: 1, color: "#f59e0b", dash: "dot" },
      name: "P90 Safety Upper Bound",
    },
    {
      x: hrs,
      y: p10,
      mode: "lines",
      line: { width: 0 },
      fill: "tonexty",
      fillcolor: "rgba(59, 130, 246, 0.22)",
      name: "P10–P90 Band",
    },
    {
      x: hrs,
      y: p50,
      mode: "lines+markers",
      line: { color: "#3b82f6", width: 2.5 },
      name: "Simulated P50 Depth (mm)",
    },
  ];

  const layout = {
    paper_bgcolor: theme.paper_bgcolor,
    plot_bgcolor: theme.plot_bgcolor,
    font: theme.font,
    hoverlabel: theme.hoverlabel,
    hovermode: "x unified",
    margin: { t: 20, r: 24, b: 45, l: 58 },
    xaxis: { title: "Hours Since Storm Onset", gridcolor: theme.gridcolor },
    yaxis: { title: "Simulated Streamway Depth (mm)", gridcolor: theme.gridcolor },
    shapes: [
      {
        type: "line",
        xref: "paper",
        x0: 0,
        x1: 1,
        yref: "y",
        y0: 700,
        y1: 700,
        line: { color: "#ef4444", width: 2, dash: "dash" },
      },
    ],
  };

  Plotly.react("sim-chart", traces, layout, { responsive: true, displayModeBar: false });
}

["sim-rain", "sim-moisture", "sim-start-depth"].forEach((id) => {
  document.getElementById(id).addEventListener("input", runSimulation);
});

document.querySelectorAll(".sim-preset").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.getElementById("sim-rain").value = btn.dataset.rain;
    document.getElementById("sim-moisture").value = btn.dataset.moist;
    document.getElementById("sim-start-depth").value = btn.dataset.depth;
    runSimulation();
  });
});

// Phase 1 & 2 Diagnostics Tab
async function loadDiagnostics() {
  const resp = await fetch("/api/diagnostics");
  latestDiagnosticsData = await resp.json();
  renderDiagnosticsCharts(latestDiagnosticsData);
}

function renderDiagnosticsCharts(diag) {
  const theme = getPlotlyTheme();
  const baseLayout = (xTitle, yTitle) => ({
    paper_bgcolor: theme.paper_bgcolor,
    plot_bgcolor: theme.plot_bgcolor,
    font: theme.font,
    hoverlabel: theme.hoverlabel,
    margin: { t: 20, r: 20, b: 45, l: 55 },
    legend: { orientation: "h", y: 1.12, x: 0 },
    xaxis: { title: xTitle, gridcolor: theme.gridcolor },
    yaxis: { title: yTitle, gridcolor: theme.gridcolor },
  });

  // 1. CCF Chart
  Plotly.react(
    "ccf-chart",
    [
      {
        x: diag.lags,
        y: diag.ccf.stump,
        type: "bar",
        name: "The Stump → ΔDepth",
        marker: { color: "#10b981" },
      },
      {
        x: diag.lags,
        y: diag.ccf.swcc,
        type: "bar",
        name: "SWCC HQ → ΔDepth",
        marker: { color: "#3b82f6" },
      },
      {
        x: diag.lags,
        y: diag.ccf.reference,
        mode: "lines+markers",
        name: "Reference Grid",
        line: { color: "#f59e0b", width: 2 },
      },
    ],
    { ...baseLayout("Lag Horizon (Hours Ahead)", "Cross-Correlation (r)"), barmode: "group" },
    { responsive: true, displayModeBar: false }
  );

  // 2. ACF Chart
  Plotly.react(
    "acf-chart",
    [
      {
        x: diag.lags,
        y: diag.acf.recession,
        mode: "lines+markers",
        name: "Recession Limb (Falling)",
        line: { color: "#10b981", width: 2.5 },
      },
      {
        x: diag.lags,
        y: diag.acf.overall,
        mode: "lines",
        name: "Overall Depth ACF",
        line: { color: "#64748b", width: 2, dash: "dash" },
      },
      {
        x: diag.lags,
        y: diag.acf.rising,
        mode: "lines+markers",
        name: "Rising Flood Pulse (>15mm/h)",
        line: { color: "#ef4444", width: 2.5 },
      },
    ],
    baseLayout("Lag Horizon (Hours Ahead)", "Autocorrelation (r)"),
    { responsive: true, displayModeBar: false }
  );

  // 3. Dry Rain Chart
  Plotly.react(
    "dry-rain-chart",
    [
      {
        x: diag.dry_rain_storms.map((s) => s.effective_rain),
        y: diag.dry_rain_storms.map((s) => s.peak_depth_mm),
        text: diag.dry_rain_storms.map(
          (s) => `${s.timestamp}<br>Raw 6h Rain: ${s.rain_6h_mm} mm<br>Moist Index: ${s.moisture_index}`
        ),
        mode: "markers",
        name: "Storms (Peat-Weighted)",
        marker: {
          size: 8,
          color: diag.dry_rain_storms.map((s) => s.moisture_index),
          colorscale: "Viridis",
          showscale: true,
          colorbar: { title: "Peat Sat" },
        },
      },
    ],
    {
      ...baseLayout("Peat-Tank Weighted 6h Effective Rainfall", "Peak Streamway Depth (mm)"),
      shapes: [
        {
          type: "line",
          xref: "paper",
          x0: 0,
          x1: 1,
          yref: "y",
          y0: 700,
          y1: 700,
          line: { color: "#ef4444", width: 2, dash: "dash" },
        },
      ],
    },
    { responsive: true, displayModeBar: false }
  );

  // 4. Horizon Breakdown (1h to 48h)
  const models = [
    ["Persistence", "#94a3b8"],
    ["Telemetry-Only (No Future Wx)", "#ef4444"],
    ["NWP-Coupled (Realistic Wx Error)", "#3b82f6"],
    ["NWP-Coupled (Exact Wx Upper Bound)", "#10b981"],
  ];
  const hTraces = models.map(([mName, col]) => {
    const pts = diag.horizon_breakdown.filter((r) => r.Model === mName);
    return {
      x: pts.map((p) => p.h_num),
      y: pts.map((p) => p.NSE),
      mode: "lines+markers",
      name: mName,
      line: { color: col, width: 2.5 },
    };
  });

  Plotly.react(
    "horizon-chart",
    hTraces,
    {
      ...baseLayout("Forecast Horizon (Hours Ahead)", "Nash-Sutcliffe Efficiency (NSE)"),
      shapes: [
        {
          type: "line",
          xref: "paper",
          x0: 0,
          x1: 1,
          yref: "y",
          y0: 0.75,
          y1: 0.75,
          line: { color: "#f59e0b", width: 2, dash: "dash" },
        },
      ],
    },
    { responsive: true, displayModeBar: false }
  );
}

document.getElementById("refresh-btn").addEventListener("click", () => fetchDashboard(true));
window.addEventListener("DOMContentLoaded", () => fetchDashboard(false));
