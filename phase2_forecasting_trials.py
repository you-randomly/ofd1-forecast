import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return Path, go, make_subplots, mo, np, pd


@app.cell
def _(mo):
    mo.md(r"""
# OFD1 Streamway: Phase 2 Forecasting Trials & Ablation Benchmark
Out-of-sample chronological evaluation (**Train:** `2024–2025` [`14,242h`], **Test:** `2026` [`7,478h`]) across **+1h, +2h, +3h, and +6h** lead times.

### Model Ablation Ladder
0. **Naive Persistence ($\hat{y}_{t+h} = y_t$):** Baseline benchmark for Nash-Sutcliffe Efficiency (NSE) and Skill over Persistence.
1. **Linear ARX (`Ridge`):** Autoregressive depth lags + raw gauge/reference rainfall lags without nonlinear soil-tank interactions.
2. **Raw GBDT (`HistGradientBoosting`):** Nonlinear tree ensemble trained on raw uncleaned gauges without antecedent peat tanks or baseflow datum normalization.
3. **Hydro-GBDT (Strictly Causal Telemetry Nowcast):** Incorporates all Phase 1 findings — Stump stuck-counter glitch filter, multi-gauge failover, high-wind under-catch scaling, 30-day 5th-percentile baseflow datum normalization, leaky-bucket Epikarst (`3–24h`) & Peat Bog (`96–168h`) soil tanks, and `P10–P90` quantile uncertainty bands.
4. **Hydro-GBDT (+Future Rain `[t+1..t+h]`):** Couples the causal Hydro-GBDT state with subsequent surface rainfall over the forecast window to quantify NWP-coupled skill at `+3h` and `+6h`.
""")
    return


@app.cell
def _(Path, mo, pd):
    metrics_path = Path("data/phase2_trial_metrics.parquet")
    preds_path = Path("data/phase2_trial_predictions.parquet")

    df_metrics = pd.read_parquet(metrics_path)
    df_preds = pd.read_parquet(preds_path)

    mo.md(
        f"**Loaded Phase 2 Trial Benchmarks:** `{len(df_metrics)}` model-horizon evaluations across `{len(df_preds) // 4:,}` out-of-sample 2026 test hours."
    )
    return df_metrics, df_preds


@app.cell
def _(df_metrics, go, make_subplots, mo):
    # Interactive Horizon Selector & Ablation Summary
    horizon_select = mo.ui.dropdown(
        options=["+1h", "+2h", "+3h", "+6h"],
        value="+2h",
        label="Select Forecast Lead Horizon",
    )

    fig_ablation = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Skill vs. Persistence (%) Across Lead Horizons",
            "Rising Flood Pulse RMSE (mm) (True ΔDepth ≥ +50 mm)",
        ),
    )

    colors = {
        "0. Naive Persistence": "#94a3b8",
        "1. Linear ARX (Raw)": "#f59e0b",
        "2. GBDT (Raw / No Soil Tank)": "#ef4444",
        "3. Hydro-GBDT (Causal Nowcast)": "#3b82f6",
        "4. Hydro-GBDT (+Future Rain)": "#10b981",
    }

    # Exclude Linear ARX from bar chart y-axis scale since its Oct 2026 Stump glitch blowup (-14,494% skill) dwarfs the scale
    plot_metrics = df_metrics[df_metrics["Trial"] != "1. Linear ARX (Raw)"]

    for trial_name, grp in plot_metrics.groupby("Trial", sort=False):
        fig_ablation.add_trace(
            go.Bar(
                x=grp["Horizon"],
                y=grp["Skill vs Pers (%)"],
                name=trial_name,
                marker_color=colors.get(trial_name, "#6366f1"),
            ),
            row=1,
            col=1,
        )
        fig_ablation.add_trace(
            go.Bar(
                x=grp["Horizon"],
                y=grp["Flood Rise RMSE (mm)"],
                name=trial_name,
                marker_color=colors.get(trial_name, "#6366f1"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig_ablation.update_yaxes(title_text="Skill Improvement over Persistence (%)", row=1, col=1)
    fig_ablation.update_yaxes(title_text="Flood Pulse RMSE (mm, Lower is Better)", row=1, col=2)
    fig_ablation.update_layout(barmode="group", height=400, margin=dict(t=45, b=40))

    mo.vstack(
        [
            mo.md("## 1. Multi-Horizon Ablation Benchmark (2026 Out-of-Sample Test Set)"),
            mo.ui.table(df_metrics),
            fig_ablation,
            horizon_select,
        ]
    )
    return (horizon_select,)


@app.cell
def _(df_preds, go, horizon_select, make_subplots, mo):
    # Hydrograph Inspection for Selected Horizon
    _h_str = horizon_select.value.replace("+", "plus_")
    _sub = df_preds.loc[_h_str].copy()

    window_select = mo.ui.dropdown(
        options={
            "Winter Flood PulseTrain (Jan–Feb 2026)": ("2026-01-15", "2026-02-15"),
            "Summer 'Dry Rain' & Late August Storm (Aug 2026)": ("2026-08-10", "2026-09-05"),
            "Autumn Floods & Stump Glitch Window (Oct 2026)": ("2026-10-01", "2026-10-25"),
            "Full 2026 Out-of-Sample Test Set": ("2026-01-01", "2026-11-08"),
        },
        value="Autumn Floods & Stump Glitch Window (Oct 2026)",
        label="Inspect 2026 Hydrograph Regime",
    )
    return (window_select,)


@app.cell
def _(df_preds, go, horizon_select, make_subplots, mo, window_select):
    _h_key = horizon_select.value.replace("+", "plus_")
    _start, _end = window_select.value
    _slice = df_preds.loc[_h_key].loc[_start:_end]

    fig_hydro = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.06,
        subplot_titles=(
            f"OFD1 Streamway Depth Forecast ({horizon_select.value} Lead) with P10–P90 Uncertainty Band",
            "Fused Surface Rainfall (mm/h) & Antecedent Peat Moisture Index",
        ),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    # P10 - P90 Shaded Quantile Band
    fig_hydro.add_trace(
        go.Scatter(
            x=_slice.index,
            y=_slice["pred_hydro_p90"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig_hydro.add_trace(
        go.Scatter(
            x=_slice.index,
            y=_slice["pred_hydro_p10"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.20)",
            name="Hydro-GBDT P10–P90 Band",
        ),
        row=1,
        col=1,
    )

    fig_hydro.add_trace(
        go.Scatter(
            x=_slice.index,
            y=_slice["actual_depth"],
            name="Actual Streamway Depth (mm)",
            line=dict(color="#0f172a", width=2.2),
        ),
        row=1,
        col=1,
    )
    fig_hydro.add_trace(
        go.Scatter(
            x=_slice.index,
            y=_slice["pred_persistence"],
            name=f"Naive Persistence ({horizon_select.value})",
            line=dict(color="#94a3b8", width=1.4, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig_hydro.add_trace(
        go.Scatter(
            x=_slice.index,
            y=_slice["pred_gbdt_raw"],
            name="Raw GBDT (Unfiltered)",
            line=dict(color="#ef4444", width=1.4, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig_hydro.add_trace(
        go.Scatter(
            x=_slice.index,
            y=_slice["pred_hydro_causal"],
            name="Hydro-GBDT (Causal Nowcast)",
            line=dict(color="#2563eb", width=2.0),
        ),
        row=1,
        col=1,
    )
    fig_hydro.add_trace(
        go.Scatter(
            x=_slice.index,
            y=_slice["pred_hydro_nwp"],
            name="Hydro-GBDT (+Future Rain)",
            line=dict(color="#10b981", width=1.8),
        ),
        row=1,
        col=1,
    )

    fig_hydro.add_hline(
        y=700,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="700 mm Amber/Red Threshold (~6\" over The Step)",
        row=1,
        col=1,
    )

    fig_hydro.add_trace(
        go.Bar(
            x=_slice.index,
            y=_slice["fused_rain_mm"],
            name="Fused Surface Rain (mm/h)",
            marker_color="#3b82f6",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig_hydro.add_trace(
        go.Scatter(
            x=_slice.index,
            y=_slice["moisture_index"],
            name="Peat Moisture Index",
            line=dict(color="#d97706", width=2),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    fig_hydro.update_yaxes(title_text="Depth (mm)", row=1, col=1)
    fig_hydro.update_yaxes(title_text="Rain (mm/h)", row=2, col=1, secondary_y=False)
    fig_hydro.update_yaxes(title_text="Moisture Index", row=2, col=1, secondary_y=True)
    fig_hydro.update_layout(height=620, margin=dict(t=45, b=40))

    mo.vstack([window_select, fig_hydro])
    return


@app.cell
def _(Path, go, make_subplots, mo, pd):
    # Section 3: Long-Horizon Breakdown Analysis (+1h to +48h)
    breakdown_path = Path("data/horizon_breakdown_1h_to_48h.parquet")
    df_long = pd.read_parquet(breakdown_path)

    fig_long = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Nash-Sutcliffe Efficiency (NSE) vs. Horizon (+1h to +48h)",
            "Overall RMSE (mm) vs. Horizon (+1h to +48h)",
        ),
    )

    long_colors = {
        "Persistence": "#94a3b8",
        "Telemetry-Only (No Future Wx)": "#ef4444",
        "NWP-Coupled (Realistic Wx Error)": "#3b82f6",
        "NWP-Coupled (Exact Wx Upper Bound)": "#10b981",
    }

    for model_name, grp in df_long.groupby("Model", sort=False):
        fig_long.add_trace(
            go.Scatter(
                x=grp["h_num"],
                y=grp["NSE"],
                mode="lines+markers",
                name=model_name,
                line=dict(color=long_colors.get(model_name, "#6366f1"), width=2.5),
            ),
            row=1,
            col=1,
        )
        fig_long.add_trace(
            go.Scatter(
                x=grp["h_num"],
                y=grp["RMSE"],
                mode="lines+markers",
                name=model_name,
                line=dict(color=long_colors.get(model_name, "#6366f1"), width=2.5),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig_long.add_hline(
        y=0.75,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="NSE = 0.75 ('Very Good' Hydrology Threshold)",
        row=1,
        col=1,
    )

    fig_long.update_xaxes(title_text="Forecast Horizon (Hours Ahead)", row=1, col=1)
    fig_long.update_xaxes(title_text="Forecast Horizon (Hours Ahead)", row=1, col=2)
    fig_long.update_yaxes(title_text="Nash-Sutcliffe Efficiency (NSE)", row=1, col=1)
    fig_long.update_yaxes(title_text="RMSE (mm, Lower is Better)", row=1, col=2)
    fig_long.update_layout(height=420, margin=dict(t=45, b=40))

    mo.vstack(
        [
            mo.md(
                """
                ## 3. Horizon Breakdown Curve (+1h to +48h): Telemetry-Only vs. NWP-Coupled
                - **Telemetry-Only Breakdown (`+6h` to `+12h`):** Without a weather forecast, the model relies solely on rain already on the ground (`Tc = 2h`) and current recession rate. It crosses below `NSE = 0.75` between **`+8h` and `+12h`** (`NSE = 0.662` at `+12h`, `0.407` at `+24h`).
                - **NWP-Coupled Endurance (`+12h` to `+48h`):** Because the 120-hour **Peat Bog Tank** saturation state is known today, driving the hydrological model forward with forecasted surface precipitation preserves **`NSE ≥ 0.80` out to `+24h`** (`MAE = 55 mm`) and **`NSE = 0.75` out to `+48h`** (`MAE = 64.5 mm`), before weather-forecast timing errors become the dominant bottleneck beyond 48 hours.
                """
            ),
            mo.ui.table(df_long.drop(columns=["h_num"])),
            fig_long,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
