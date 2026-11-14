from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

HORIZONS = [1, 2, 3, 6, 12, 18, 24, 36, 48]
EVAL_SKILL_HORIZONS = [1, 3, 6, 12, 18, 24, 36, 48]
STEP_DATUM_MM = 550.0
ELEVATED_DEPTH_MM = 600.0
AMBER_DEPTH_MM = 620.0
RED_DEPTH_MM = 700.0  # ~6 inches over The Step
RED_TRIGGER_DEPTH_MM = 705.0  # Rising limb trigger threshold (prevents edge-touching false alarms)
RED_CLEAR_DEPTH_MM = 675.0  # Falling limb clearance threshold (hysteresis deadband)
MIN_PHYSICAL_DEPTH_MM = 350.0
TRAIN_SPLIT_UTC = pd.Timestamp("2026-01-01 00:00:00+00:00")

NWP_MODELS = [
    "best_match",
    "ukmo_seamless",
    "ecmwf_ifs025",
    "icon_seamless",
    "gfs_seamless",
    "meteofrance_seamless",
]

NWP_MEMBER_MAP = {
    "nwp_ukmo_seamless": "UKMO 2km UKV",
    "nwp_ecmwf_ifs025": "ECMWF IFS 0.25°",
    "nwp_icon_seamless": "DWD ICON-EU",
    "nwp_meteofrance_seamless": "Météo-France AROME",
    "nwp_gfs_seamless": "NOAA GFS",
    "nwp_best_match": "Open-Meteo Blend",
}

CAUSAL_COLS = (
    [
        "depth_mm",
        "stage_above_base",
        "baseflow_30d",
        "ddepth_1h",
        "ddepth_2h",
        "ddepth_3h",
        "ddepth_4h",
        "ddepth_6h",
        "ddepth_12h",
        "ddepth_24h",
        "accel_1h",
        "moisture_index",
        "era5_soil_moisture",
        "era5_soil_deep",
        "tank_3h",
        "tank_6h",
        "tank_12h",
        "tank_24h",
        "tank_96h",
        "tank_168h",
        "ref_tank_6h",
        "ref_tank_96h",
        "wind_kmh",
        "temp_c",
        "press_hpa",
        "dpress_3h",
        "dpress_6h",
    ]
    + [f"fused_rain_lag_{k}h" for k in range(7)]
    + [f"ref_rain_lag_{k}h" for k in range(7)]
    + [f"eff_rain_lag_{k}h" for k in range(7)]
    + [f"fused_rain_sum_{w}h" for w in [2, 3, 6, 12, 24]]
    + [f"ref_rain_sum_{w}h" for w in [2, 3, 6, 12, 24]]
    + [f"eff_rain_sum_{w}h" for w in [2, 3, 6, 12, 24]]
)


def nwp_cols_for_horizon(h: int) -> list[str]:
    cols = [
        "fut_ref_tank_6h",
        "fut_ref_tank_24h",
        "fut_ref_tank_96h",
        "fut_rain_sum_6h_at_target",
        "fut_rain_sum_12h_at_target",
        "fut_eff_rain_sum_6h_at_target",
    ]
    for k in range(min(h, 6)):
        cols.append(f"fut_rain_minus_{k}h_from_target")
    return cols


def engineer_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Compute causal hydrological features, glitch filters, and full-history expanding bounds."""
    df = df_in.copy().sort_index()

    # 1. Depth lags, velocity & non-flood step-drop momentum clamp
    for lag in [1, 2, 3, 4, 6, 12, 24]:
        df[f"ddepth_{lag}h"] = df["depth_mm"] - df["depth_mm"].shift(lag)
    df["accel_1h"] = df["ddepth_1h"] - df["ddepth_1h"].shift(1)

    prev_depth_2h = df["depth_mm"].shift(2)
    step_drop_hit = (prev_depth_2h < 750.0) & (
        (df["ddepth_1h"] <= -30.0) | (df["ddepth_2h"] <= -50.0)
    )
    step_drop_window = step_drop_hit.rolling(3, min_periods=1).max().fillna(0).astype(bool)
    df["depth_step_drop_flag"] = step_drop_hit.fillna(False)
    for lag in [1, 2, 3, 4, 6, 12, 24]:
        col = f"ddepth_{lag}h"
        df[col] = np.where(step_drop_window, np.maximum(df[col], -10.0), df[col])
    df["accel_1h"] = np.where(step_drop_window, np.maximum(df["accel_1h"], -5.0), df["accel_1h"])

    # 2. Glitch-filtered Stump, SWCC, and optional NRW rain with stuck-zero quarantine
    stump_glitch = (df["stump_rain_mm"] >= 15.0) & (
        (df["swcc_rain_mm"].fillna(0) < 5.0) & (df["rain_mm"].fillna(0) < 5.0)
    )
    df["stump_glitch_flag"] = stump_glitch
    stump_clean = df["stump_rain_mm"].mask(stump_glitch, np.nan)
    swcc_clean = df["swcc_rain_mm"].copy()

    nrw_raw = (
        pd.to_numeric(df["nrw_rain_mm"], errors="coerce")
        if "nrw_rain_mm" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=float)
    )
    nrw_glitch = (nrw_raw >= 25.0) & (
        (df["swcc_rain_mm"].fillna(0.0) < 5.0)
        & (stump_clean.fillna(0.0) < 5.0)
        & (df["rain_mm"].fillna(0.0) < 5.0)
    )
    df["nrw_glitch_flag"] = nrw_glitch.fillna(False)
    nrw_clean = nrw_raw.mask(nrw_glitch, np.nan)

    def _stuck_zero_mask(series: pd.Series, peer: pd.Series) -> pd.Series:
        flat_6h = series.notna() & (series.rolling(6, min_periods=6).sum() == 0.0)
        peer_rain = peer.fillna(0.0).rolling(3, min_periods=1).sum() >= 0.5
        hydro_nwp_agree = (df["ddepth_3h"].fillna(0.0) >= 10.0) & (
            df["rain_mm"].fillna(0.0).rolling(6, min_periods=1).sum() >= 1.0
        )
        return flat_6h & (peer_rain | hydro_nwp_agree)

    stump_peer = pd.concat([swcc_clean, nrw_clean], axis=1).max(axis=1)
    swcc_peer = pd.concat([stump_clean, nrw_clean], axis=1).max(axis=1)
    nrw_peer = pd.concat([stump_clean, swcc_clean], axis=1).max(axis=1)

    stump_stuck = _stuck_zero_mask(stump_clean, stump_peer)
    swcc_stuck = _stuck_zero_mask(swcc_clean, swcc_peer)
    nrw_stuck = _stuck_zero_mask(nrw_clean, nrw_peer)
    df["stump_stuck_zero_flag"] = stump_stuck.fillna(False)
    df["swcc_stuck_zero_flag"] = swcc_stuck.fillna(False)
    df["nrw_stuck_zero_flag"] = nrw_stuck.fillna(False)

    stump_clean = stump_clean.mask(stump_stuck, np.nan)
    swcc_clean = swcc_clean.mask(swcc_stuck, np.nan)
    nrw_clean = nrw_clean.mask(nrw_stuck, np.nan)
    df["stump_rain_clean"] = stump_clean
    df["nrw_rain_clean"] = nrw_clean

    df["stump_status"] = np.where(
        df["stump_rain_mm"].isna(),
        "offline",
        np.where(stump_glitch, "glitch", np.where(stump_stuck, "stuck_zero", "ok")),
    )
    df["swcc_status"] = np.where(
        df["swcc_rain_mm"].isna(),
        "offline",
        np.where(swcc_stuck, "stuck_zero", "ok"),
    )
    df["nrw_status"] = np.where(
        nrw_raw.isna(),
        "offline",
        np.where(nrw_glitch, "glitch", np.where(nrw_stuck, "stuck_zero", "ok")),
    )

    df["club_rain_mean"] = pd.concat([stump_clean, swcc_clean], axis=1).mean(axis=1)
    df["rain_source"] = np.where(
        stump_clean.notna() & swcc_clean.notna(),
        "club",
        np.where(
            swcc_clean.notna(),
            "swcc",
            np.where(
                stump_clean.notna(),
                "stump",
                np.where(nrw_clean.notna(), "nrw", "open_meteo"),
            ),
        ),
    )
    df["fused_rain_mm"] = (
        df["club_rain_mean"]
        .fillna(nrw_clean)
        .fillna(df["rain_mm"])
        .fillna(0.0)
    )

    wind_factor = 1.0 + 0.004 * (df["wind_kmh"].fillna(15.0) - 15.0).clip(lower=0.0)
    df["fused_rain_wind_adj"] = df["fused_rain_mm"] * wind_factor

    # 3. Rolling 30-day (720h) 5th-percentile baseflow datum
    df["baseflow_30d"] = df["depth_mm"].rolling(720, min_periods=24).quantile(0.05).bfill()
    df["stage_above_base"] = df["depth_mm"] - df["baseflow_30d"]

    # 4. Antecedent moisture tanks
    et0 = df["et0_mm"].fillna(0.05)
    net_rain = (df["fused_rain_mm"] - 0.8 * et0).clip(lower=0.0)
    rain_ref = df["rain_mm"].fillna(df["fused_rain_mm"]).fillna(0.0)

    for hl in [3, 6, 12, 24, 96, 168]:
        alpha = 1.0 - np.exp(-np.log(2) / hl)
        df[f"tank_{hl}h"] = net_rain.ewm(alpha=alpha, adjust=False).mean()
        df[f"ref_tank_{hl}h"] = rain_ref.ewm(alpha=alpha, adjust=False).mean()

    soil_min = df["era5_soil_moisture"].expanding(min_periods=48).min().fillna(0.30)
    soil_max = df["era5_soil_moisture"].expanding(min_periods=48).max().fillna(0.43)
    tank_168h_p95 = df["tank_168h"].expanding(min_periods=48).quantile(0.95).fillna(0.5)
    df["soil_min"] = soil_min
    df["soil_max"] = soil_max
    df["tank_168h_p95"] = tank_168h_p95

    sat_norm = ((df["era5_soil_moisture"].ffill() - soil_min) / (soil_max - soil_min + 1e-6)).clip(
        0.05, 1.0
    )
    tank_norm = (df["tank_168h"] / (tank_168h_p95 + 1e-6)).clip(0.05, 1.5)
    df["moisture_index"] = (0.5 * sat_norm + 0.5 * tank_norm).fillna(0.5)
    df["eff_rain_mm"] = df["fused_rain_wind_adj"] * (df["moisture_index"] ** 1.5)

    # 5. Rain lags & rolling sums
    for lag in range(7):
        df[f"fused_rain_lag_{lag}h"] = df["fused_rain_wind_adj"].shift(lag)
        df[f"ref_rain_lag_{lag}h"] = rain_ref.shift(lag)
        df[f"eff_rain_lag_{lag}h"] = df["eff_rain_mm"].shift(lag)

    for win in [2, 3, 6, 12, 24]:
        df[f"fused_rain_sum_{win}h"] = df["fused_rain_wind_adj"].rolling(win).sum()
        df[f"ref_rain_sum_{win}h"] = rain_ref.rolling(win).sum()
        df[f"eff_rain_sum_{win}h"] = df["eff_rain_mm"].rolling(win).sum()

    df["press_hpa"] = (
        df["swcc_press_hpa"].fillna(df["stump_press_hpa"]).fillna(df["era5_press_hpa"])
    )
    df["dpress_3h"] = df["press_hpa"] - df["press_hpa"].shift(3)
    df["dpress_6h"] = df["press_hpa"] - df["press_hpa"].shift(6)

    return df


def attach_horizon_future_cols(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Attach target-time future rainfall and peat-tank features for horizon ``h``."""
    work = df.copy()
    work["fut_ref_tank_6h"] = work["ref_tank_6h"].shift(-h)
    work["fut_ref_tank_24h"] = work["tank_24h"].shift(-h)
    work["fut_ref_tank_96h"] = work["ref_tank_96h"].shift(-h)
    work["fut_rain_sum_6h_at_target"] = work["ref_rain_sum_6h"].shift(-h)
    work["fut_rain_sum_12h_at_target"] = work["ref_rain_sum_12h"].shift(-h)
    work["fut_eff_rain_sum_6h_at_target"] = work["eff_rain_sum_6h"].shift(-h)
    for k in range(min(h, 6)):
        work[f"fut_rain_minus_{k}h_from_target"] = work["rain_mm"].shift(-(h - k))
    return work


def build_archived_nwp_trajectories(
    arch: pd.DataFrame, valid_obs: pd.DataFrame, issue_idx: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (N, 48) matrices of future hourly rain from archived NWP d0/d1/d2 forecasts:
    (det_rain, ens_p10, ens_p50, ens_p90, perf_rain, member_spread_48h).
    """
    n_issues = len(issue_idx)
    full_idx = pd.date_range(
        issue_idx.min() + pd.Timedelta(hours=1),
        issue_idx.max() + pd.Timedelta(hours=48),
        freq="1h",
        tz="UTC",
    )
    arch_re = arch.reindex(full_idx)
    obs_rain_s = valid_obs["rain_mm"].reindex(full_idx).fillna(0.0).values

    base_offsets = ((issue_idx - full_idx[0]) // pd.Timedelta(hours=1)).values.astype(int)
    grid_idx = base_offsets[:, None] + np.arange(48, dtype=int)[None, :]

    lead_h = np.arange(1, 49)
    m_d0 = lead_h <= 6
    m_d1 = (lead_h > 6) & (lead_h <= 30)
    m_d2 = lead_h > 30

    member_mats = []
    for slug in NWP_MODELS:
        c0 = arch_re[f"{slug}_d0"].values if f"{slug}_d0" in arch_re else np.full(len(full_idx), np.nan)
        c1 = arch_re[f"{slug}_d1"].values if f"{slug}_d1" in arch_re else np.full(len(full_idx), np.nan)
        c2 = arch_re[f"{slug}_d2"].values if f"{slug}_d2" in arch_re else np.full(len(full_idx), np.nan)
        mat = np.empty((n_issues, 48), dtype=float)
        mat[:, m_d0] = c0[grid_idx[:, m_d0]]
        mat[:, m_d1] = c1[grid_idx[:, m_d1]]
        mat[:, m_d2] = c2[grid_idx[:, m_d2]]
        member_mats.append(mat)

    stack = np.stack(member_mats, axis=0)
    det_rain = np.where(np.isnan(stack[0]), np.nanmedian(stack, axis=0), stack[0])
    det_rain = np.nan_to_num(det_rain, nan=0.0)

    rain_p10, rain_p50, rain_p90, spread_48h = compute_nwp_ensemble_quantiles(stack, det_rain)
    perf_rain = obs_rain_s[grid_idx]
    return det_rain, rain_p10, rain_p50, rain_p90, perf_rain, spread_48h


def roll_future_features_for_horizon(
    valid_obs: pd.DataFrame,
    issue_idx: pd.DatetimeIndex,
    fut_rain_mat: np.ndarray,
    h: int,
) -> pd.DataFrame:
    """Compute exact CAUSAL_COLS + nwp_cols_for_horizon(h) at issue_idx for an (N, >=h) future rain matrix."""
    sub = valid_obs.loc[issue_idx, CAUSAL_COLS].copy()
    full_idx = valid_obs.index
    base_pos = full_idx.get_indexer(issue_idx)

    rain_obs_all = valid_obs["rain_mm"].fillna(valid_obs["fused_rain_mm"]).fillna(0.0).values
    eff_obs_all = valid_obs["eff_rain_mm"].fillna(0.0).values
    et0_t0 = valid_obs.loc[issue_idx, "et0_mm"].fillna(0.05).values
    wind_t0 = valid_obs.loc[issue_idx, "wind_kmh"].fillna(15.0).values
    wind_factor = 1.0 + 0.004 * np.clip(wind_t0 - 15.0, 0.0, None)
    moist_t0 = valid_obs.loc[issue_idx, "moisture_index"].fillna(0.5).values

    ref_tank_6 = valid_obs.loc[issue_idx, "ref_tank_6h"].values.astype(float).copy()
    tank_24 = valid_obs.loc[issue_idx, "tank_24h"].values.astype(float).copy()
    ref_tank_96 = valid_obs.loc[issue_idx, "ref_tank_96h"].values.astype(float).copy()

    a6 = 1.0 - np.exp(-np.log(2) / 6.0)
    a24 = 1.0 - np.exp(-np.log(2) / 24.0)
    a96 = 1.0 - np.exp(-np.log(2) / 96.0)

    for step in range(h):
        r_step = fut_rain_mat[:, step]
        net_step = np.clip(r_step - 0.8 * et0_t0, 0.0, None)
        ref_tank_6 = (1.0 - a6) * ref_tank_6 + a6 * r_step
        tank_24 = (1.0 - a24) * tank_24 + a24 * net_step
        ref_tank_96 = (1.0 - a96) * ref_tank_96 + a96 * r_step

    sub["fut_ref_tank_6h"] = ref_tank_6
    sub["fut_ref_tank_24h"] = tank_24
    sub["fut_ref_tank_96h"] = ref_tank_96

    past_12 = np.column_stack(
        [rain_obs_all[np.maximum(0, base_pos - k)] for k in range(11, -1, -1)]
    )
    comb_rain = np.hstack([past_12, fut_rain_mat[:, :h]])
    tgt_col = 11 + h
    sub["fut_rain_sum_6h_at_target"] = np.sum(comb_rain[:, tgt_col - 5 : tgt_col + 1], axis=1)
    sub["fut_rain_sum_12h_at_target"] = np.sum(comb_rain[:, tgt_col - 11 : tgt_col + 1], axis=1)

    fut_eff_mat = (fut_rain_mat[:, :h] * wind_factor[:, None]) * (moist_t0[:, None] ** 1.5)
    past_6_eff = np.column_stack(
        [eff_obs_all[np.maximum(0, base_pos - k)] for k in range(5, -1, -1)]
    )
    comb_eff = np.hstack([past_6_eff, fut_eff_mat])
    eff_tgt_col = 5 + h
    sub["fut_eff_rain_sum_6h_at_target"] = np.sum(
        comb_eff[:, eff_tgt_col - 5 : eff_tgt_col + 1], axis=1
    )

    for k in range(min(h, 6)):
        sub[f"fut_rain_minus_{k}h_from_target"] = fut_rain_mat[:, h - 1 - k]

    return sub[CAUSAL_COLS + nwp_cols_for_horizon(h)]


def roll_future_moisture_series(now_row: pd.Series | dict[str, Any], fut_wx: pd.DataFrame) -> pd.Series:
    """Roll tank_168h over future weather hours and normalize against full-history expanding bounds at t0."""
    n_hrs = len(fut_wx)
    if n_hrs == 0:
        return pd.Series([], index=fut_wx.index, dtype=float, name="moisture_index")

    rain_col = "rain_p50_mm" if "rain_p50_mm" in fut_wx.columns else "rain_mm"
    fut_rain = (
        pd.to_numeric(fut_wx[rain_col], errors="coerce").fillna(0.0).values
        if rain_col in fut_wx.columns
        else np.zeros(n_hrs, dtype=float)
    )
    fut_et0 = (
        pd.to_numeric(fut_wx["et0_mm"], errors="coerce").fillna(0.05).values
        if "et0_mm" in fut_wx.columns
        else np.full(n_hrs, float(now_row.get("et0_mm", 0.05)), dtype=float)
    )
    fut_soil = (
        pd.to_numeric(fut_wx["era5_soil_moisture"], errors="coerce")
        .fillna(float(now_row.get("era5_soil_moisture", 0.39)))
        .values
        if "era5_soil_moisture" in fut_wx.columns
        else np.full(n_hrs, float(now_row.get("era5_soil_moisture", 0.39)), dtype=float)
    )

    a168 = 1.0 - np.exp(-np.log(2) / 168.0)
    t168 = float(now_row.get("tank_168h", 0.5))
    tank_168_vals = np.empty(n_hrs, dtype=float)
    for step in range(n_hrs):
        net_step = max(0.0, float(fut_rain[step]) - 0.8 * float(fut_et0[step]))
        t168 = (1.0 - a168) * t168 + a168 * net_step
        tank_168_vals[step] = t168

    s_min = float(now_row.get("soil_min", 0.30))
    s_max = float(now_row.get("soil_max", 0.43))
    t95 = float(now_row.get("tank_168h_p95", 0.5))
    if np.isnan(s_min):
        s_min = 0.30
    if np.isnan(s_max):
        s_max = 0.43
    if np.isnan(t95) or t95 <= 0.0:
        t95 = 0.5

    sat_norm = np.clip((fut_soil - s_min) / (s_max - s_min + 1e-6), 0.05, 1.0)
    tank_norm = np.clip(tank_168_vals / (t95 + 1e-6), 0.05, 1.5)
    moist = 0.5 * sat_norm + 0.5 * tank_norm
    moist = np.where(np.isnan(moist), 0.5, moist)
    return pd.Series(moist, index=fut_wx.index, dtype=float, name="moisture_index")


def compute_nwp_ensemble_quantiles(
    raw_stack: np.ndarray, det_rain: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute volume-conserved NWP ensemble P10, P50, P90 rain and 48h spread.

    Evaluates the 2-model wet quorum (>= 0.6 mm/h) on ``raw_stack`` BEFORE filling missing
    (NaN) member values with ``det_rain``. Supports both 2D ``(M, T)`` and 3D ``(M, N, T)`` inputs.
    """
    raw_arr = np.asarray(raw_stack, dtype=float)
    det_arr = np.nan_to_num(np.asarray(det_rain, dtype=float), nan=0.0)

    # Evaluate wet quorum on raw un-imputed member predictions
    wet_members_count = np.sum((~np.isnan(raw_arr)) & (raw_arr >= 0.6), axis=0)

    # Fill missing member values with det_rain
    stack_filled = np.where(np.isnan(raw_arr), np.expand_dims(det_arr, axis=0), raw_arr)

    ens_min = np.percentile(stack_filled, 10, axis=0)
    ens_med = np.percentile(stack_filled, 50, axis=0)
    raw_ens_max = np.percentile(stack_filled, 90, axis=0)

    ens_max = np.where(
        wet_members_count >= 2,
        raw_ens_max,
        np.minimum(raw_ens_max, 0.5 * (ens_med + raw_ens_max)),
    )

    rain_p50 = np.round(0.5 * det_arr + 0.5 * ens_med, 2)
    rain_p10 = np.round(np.minimum(rain_p50, ens_min), 2)

    member_totals_48h = np.sum(stack_filled[..., :48], axis=-1)
    high_vol_count = np.sum(member_totals_48h >= 10.0, axis=0)
    target_p90_raw = np.percentile(member_totals_48h, 90, axis=0)
    target_p75_raw = np.percentile(member_totals_48h, 75, axis=0)
    target_p90_total = np.where(high_vol_count >= 2, target_p90_raw, target_p75_raw)

    raw_p90 = np.maximum(rain_p50, ens_max)
    raw_p90_sum = np.sum(raw_p90[..., :48], axis=-1)

    vol_scale = np.where(
        (raw_p90_sum > 1e-3) & (target_p90_total > 0.0),
        np.clip(target_p90_total / np.maximum(raw_p90_sum, 1e-6), 0.55, 1.15),
        1.0,
    )
    rain_p90 = np.round(
        np.maximum(rain_p50, raw_p90 * np.expand_dims(vol_scale, axis=-1)), 2
    )
    spread_48h = np.max(member_totals_48h, axis=0) - np.min(member_totals_48h, axis=0)

    return rain_p10, rain_p50, rain_p90, spread_48h


def enrich_nwp_ensemble_quantiles(
    fut_wx: pd.DataFrame,
    now_row: pd.Series | dict[str, Any],
    use_ensemble: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Enrich a single-anchor 48h future weather DataFrame with P10/P50/P90 rain trajectories and metadata."""
    df_w = fut_wx.copy()
    n_hrs = len(df_w)
    lead_hours = np.arange(1, n_hrs + 1, dtype=float)
    lead_factor = np.clip((lead_hours - 2.0) / 24.0, 0.05, 1.50)

    det_rain = (
        pd.to_numeric(df_w["rain_mm"], errors="coerce").fillna(0.0).values
        if "rain_mm" in df_w.columns
        else np.zeros(n_hrs, dtype=float)
    )
    if not use_ensemble:
        det = np.round(det_rain, 2)
        df_w["rain_p10_mm"] = det
        df_w["rain_p50_mm"] = det
        df_w["rain_p90_mm"] = det
        df_w["rain_mm"] = det
        if "precip_prob_pct" not in df_w.columns:
            df_w["precip_prob_pct"] = 0.0
        df_w["precip_prob_pct"] = pd.to_numeric(df_w["precip_prob_pct"], errors="coerce").fillna(
            0.0
        )
        meta = {
            "models_count": 1,
            "mode": "Deterministic NWP (hydrological uncertainty only)",
            "rain_24h_p10_mm": round(float(det[:24].sum()), 1),
            "rain_24h_p50_mm": round(float(det[:24].sum()), 1),
            "rain_24h_p90_mm": round(float(det[:24].sum()), 1),
            "rain_48h_p10_mm": round(float(det[:48].sum()), 1),
            "rain_48h_p50_mm": round(float(det[:48].sum()), 1),
            "rain_48h_p90_mm": round(float(det[:48].sum()), 1),
            "max_precip_prob_pct": round(float(df_w["precip_prob_pct"].max()) if n_hrs else 0),
            "members": [],
        }
        return df_w, meta

    prob_col = (
        "ens_prob_max_pct"
        if "ens_prob_max_pct" in df_w.columns
        else ("precip_prob_pct" if "precip_prob_pct" in df_w.columns else None)
    )
    prob_pct = (
        pd.to_numeric(df_w[prob_col], errors="coerce").fillna(0.0).values
        if prob_col
        else np.zeros(n_hrs, dtype=float)
    )

    avail_cols = [c for c in NWP_MEMBER_MAP if c in df_w.columns and df_w[c].notna().any()]

    if avail_cols:
        raw_stack = np.vstack(
            [pd.to_numeric(df_w[c], errors="coerce").values.astype(float) for c in avail_cols]
        )
        rain_p10, rain_p50, rain_p90, _ = compute_nwp_ensemble_quantiles(raw_stack, det_rain)
        members_summary = [
            {
                "id": col.replace("nwp_", ""),
                "label": NWP_MEMBER_MAP[col],
                "rain_24h_mm": round(
                    float(
                        pd.to_numeric(df_w[col], errors="coerce").fillna(0.0).iloc[:24].sum()
                    ),
                    1,
                ),
                "rain_48h_mm": round(
                    float(
                        pd.to_numeric(df_w[col], errors="coerce").fillna(0.0).iloc[:48].sum()
                    ),
                    1,
                ),
                "peak_1h_mm": round(
                    float(pd.to_numeric(df_w[col], errors="coerce").fillna(0.0).max()), 2
                ),
            }
            for col in avail_cols
        ]
    else:
        press_now = float(now_row.get("press_hpa", 985.0))
        low_press_boost = np.clip((1002.0 - press_now) / 25.0, 0.0, 1.0)
        rain_p50 = np.round(det_rain, 2)
        rain_p10 = np.round(det_rain * np.clip(1.0 - 0.45 * np.sqrt(lead_factor), 0.2, 0.85), 2)
        synoptic_tail = np.where(
            lead_hours >= 6,
            (0.12 * low_press_boost + 0.45 * det_rain) * np.sqrt(lead_factor),
            0.25 * det_rain,
        )
        rain_p90 = np.round(rain_p50 + synoptic_tail, 2)
        prob_pct = np.where(rain_p90 > 0.1, np.clip(rain_p90 * 45.0, 15.0, 95.0), 10.0)
        members_summary = []

    df_w["rain_p10_mm"] = rain_p10
    df_w["rain_p50_mm"] = rain_p50
    df_w["rain_p90_mm"] = rain_p90
    df_w["rain_mm"] = rain_p50
    df_w["precip_prob_pct"] = np.round(prob_pct, 0)

    meta = {
        "models_count": len(avail_cols) if avail_cols else 1,
        "mode": "6-Model NWP Super-Ensemble (UKMO 2km • ECMWF • ICON • AROME • GFS)"
        if avail_cols
        else "Synoptic NWP Spread Fallback",
        "rain_24h_p10_mm": round(float(rain_p10[:24].sum()), 1),
        "rain_24h_p50_mm": round(float(rain_p50[:24].sum()), 1),
        "rain_24h_p90_mm": round(float(rain_p90[:24].sum()), 1),
        "rain_48h_p10_mm": round(float(rain_p10[:48].sum()), 1),
        "rain_48h_p50_mm": round(float(rain_p50[:48].sum()), 1),
        "rain_48h_p90_mm": round(float(rain_p90[:48].sum()), 1),
        "max_precip_prob_pct": round(float(np.nanmax(prob_pct)) if len(prob_pct) else 0),
        "members": members_summary,
    }
    return df_w, meta


def combine_ensemble_quantile_deltas(
    d_p50: np.ndarray | float,
    d_p10_on_p50: np.ndarray | float,
    d_p50_on_p10: np.ndarray | float,
    d_p90_on_p50: np.ndarray | float,
    d_p90_on_p90: np.ndarray | float,
    d_p50_on_p90: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    """Combine hydrological quantile spread and NWP trajectory spread in quadrature.

    Returns ``(d_p10_rel_base, d_p50_rel_base, d_p90_rel_base, delta_hydro_high)``, where all
    three quantile deltas are in the same reference frame (depth delta in mm relative to ``base_depth``).
    """
    is_scalar = np.isscalar(d_p50) or isinstance(d_p50, float)
    dp50 = np.asarray(d_p50, dtype=float)
    dp10_p50 = np.asarray(d_p10_on_p50, dtype=float)
    dp50_p10 = np.asarray(d_p50_on_p10, dtype=float)
    dp90_p50 = np.asarray(d_p90_on_p50, dtype=float)
    dp90_p90 = np.asarray(d_p90_on_p90, dtype=float)
    dp50_p90 = np.asarray(d_p50_on_p90, dtype=float)

    delta_hydro_low = np.maximum(0.0, dp50 - dp10_p50)
    delta_nwp_low = np.maximum(0.0, dp50 - dp50_p10)
    total_low_spread = np.sqrt(delta_hydro_low**2 + delta_nwp_low**2)

    delta_hydro_high = np.maximum(0.0, dp90_p50 - dp50)
    delta_nwp_high = np.maximum(
        0.0, np.maximum(dp50_p90 - dp50, 0.35 * (dp90_p90 - dp90_p50))
    )
    total_high_spread = np.sqrt(delta_hydro_high**2 + delta_nwp_high**2)

    d_p10_rel = dp50 - total_low_spread
    d_p50_rel = dp50
    d_p90_rel = dp50 + total_high_spread

    if is_scalar:
        return (
            float(d_p10_rel),
            float(d_p50_rel),
            float(d_p90_rel),
            float(delta_hydro_high),
        )
    return d_p10_rel, d_p50_rel, d_p90_rel, delta_hydro_high


def apply_hydrological_guardrails(
    base_depth: np.ndarray | float,
    ddepth_1h: np.ndarray | float,
    fused_rain_3h: np.ndarray | float,
    moisture_index: np.ndarray | float,
    tank_168h: np.ndarray | float,
    horizon_h: int,
    fut_rain_sum: np.ndarray | float,
    d_p10: np.ndarray | float,
    d_p50: np.ndarray | float,
    d_p90: np.ndarray | float,
    obs_rain_confirmed: np.ndarray | float | bool = True,
    return_telemetry: bool = False,
) -> tuple[Any, ...]:
    """Apply physical karst rate-of-rise and peat sponge infiltration guardrails.

    At short horizons (``horizon_h <= 6``), P50 ``allowed_rise`` is capped using confirmed
    observed rain (``fused_rain_3h`` when local gauges are active) plus active stream rise
    (``ddepth_1h``), preventing unconfirmed NWP drizzle from triggering RED P50 alarms while
    still letting ``d_p90`` reflect ``fused_rain_3h + fut_rain_sum`` for AMBER watches.
    """
    is_scalar = np.isscalar(base_depth) or isinstance(base_depth, float)
    b_d = np.asarray(base_depth, dtype=float)
    dd1 = np.asarray(ddepth_1h, dtype=float)
    r3h = np.maximum(0.0, np.asarray(fused_rain_3h, dtype=float))
    rfut = np.maximum(0.0, np.asarray(fut_rain_sum, dtype=float))
    m_idx = np.asarray(moisture_index, dtype=float)
    t168 = np.asarray(tank_168h, dtype=float)
    dp10 = np.asarray(d_p10, dtype=float)
    dp50 = np.asarray(d_p50, dtype=float)
    dp90 = np.asarray(d_p90, dtype=float)
    obs_conf = np.asarray(obs_rain_confirmed, dtype=bool)

    sat_factor = np.clip(m_idx**1.5, 0.15, 1.0)
    sponge_deduct = np.where(t168 < 0.5, 8.0, 0.0)

    # P90 (and P50 for h > 6h) can respond to NWP future rain
    rain_avail_nwp = r3h + rfut
    eff_rain_nwp = np.maximum(0.0, rain_avail_nwp - sponge_deduct) * sat_factor
    max_phys_rise_nwp = 25.0 + 12.0 * eff_rain_nwp
    max_rate_rise_nwp = np.where(eff_rain_nwp < 20.0, np.maximum(35.0, 30.0 * horizon_h), 9999.0)
    allowed_rise_nwp = np.minimum(max_phys_rise_nwp, max_rate_rise_nwp)

    # Short-range (+1..+6h) P50 ignores unconfirmed NWP drizzle (<10mm) or NWP when gauges are offline
    if horizon_h <= 6:
        r3h_obs = np.where(obs_conf, r3h, 0.0)
        rfut_p50 = np.where(obs_conf & ((r3h_obs > 0.5) | (rfut >= 10.0)), rfut, 0.0)
        eff_rain_obs = np.maximum(0.0, r3h_obs + rfut_p50 - sponge_deduct) * sat_factor
        max_phys_rise_obs = 25.0 + 12.0 * eff_rain_obs + np.maximum(0.0, dd1) * 1.5
        max_rate_rise_obs = np.where(eff_rain_obs < 20.0, np.maximum(35.0, 30.0 * horizon_h), 9999.0)
        allowed_rise_p50 = np.minimum(max_phys_rise_obs, max_rate_rise_obs)
    else:
        allowed_rise_p50 = allowed_rise_nwp

    # Apply clamp only when base depth is below Amber (< 620 mm) and not already surging (ddepth_1h < 15 mm)
    clamp_mask = (b_d < AMBER_DEPTH_MM) & (dd1 < 15.0)

    clamped_p50 = np.where(clamp_mask, np.minimum(dp50, allowed_rise_p50), dp50)
    clamped_p90 = np.where(clamp_mask, np.minimum(dp90, allowed_rise_nwp * 1.5 + 20.0), dp90)
    p50_clamp_shift = dp50 - clamped_p50
    p90_clamp_shift = dp90 - clamped_p90
    clamped_p10 = np.where(
        clamp_mask,
        np.minimum(dp10 - p50_clamp_shift, clamped_p50),
        np.minimum(dp10, clamped_p50),
    )

    if return_telemetry:
        telemetry = {
            "guardrail_clamped": bool(np.any(p50_clamp_shift > 0) or np.any(p90_clamp_shift > 0)),
            "p50_clamp_shift_mm": float(np.max(p50_clamp_shift)),
            "p90_clamp_shift_mm": float(np.max(p90_clamp_shift)),
        }
        if is_scalar:
            return float(clamped_p10), float(clamped_p50), float(clamped_p90), telemetry
        return clamped_p10, clamped_p50, clamped_p90, telemetry

    if is_scalar:
        return float(clamped_p10), float(clamped_p50), float(clamped_p90)
    return clamped_p10, clamped_p50, clamped_p90


def reconstruct_depth_quantiles(
    base_depth: np.ndarray | float,
    d_p10_c: np.ndarray | float,
    d_p50_c: np.ndarray | float,
    d_p90_c: np.ndarray | float,
    delta_hydro_high: np.ndarray | float | None = None,
    min_depth_mm: float = MIN_PHYSICAL_DEPTH_MM,
) -> (
    tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]
    | tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float, np.ndarray | float]
):
    """Reconstruct monotonic absolute depth quantiles (p10, p50, p90[, p90_hydro_only]) from base_depth deltas."""
    is_scalar = np.isscalar(base_depth) or isinstance(base_depth, float)
    b_d = np.asarray(base_depth, dtype=float)
    dp10 = np.asarray(d_p10_c, dtype=float)
    dp50 = np.asarray(d_p50_c, dtype=float)
    dp90 = np.asarray(d_p90_c, dtype=float)

    p50 = np.maximum(min_depth_mm, b_d + dp50)
    p10 = np.maximum(min_depth_mm, np.minimum(p50, b_d + dp10))
    p90 = np.maximum(p50, b_d + dp90)

    if delta_hydro_high is not None:
        d_hydro = np.asarray(delta_hydro_high, dtype=float)
        p90_hydro_only = np.maximum(p50, np.minimum(p90, b_d + dp50 + d_hydro))
        if is_scalar:
            return float(p10), float(p50), float(p90), float(p90_hydro_only)
        return p10, p50, p90, p90_hydro_only

    if is_scalar:
        return float(p10), float(p50), float(p90)
    return p10, p50, p90


def compute_initial_alert_status(
    base_depth: float,
    recent_depths_mm: pd.Series | np.ndarray | list[float] | None = None,
) -> str:
    """Determine the initial RED/GREEN hysteresis state at issue time t0.

    Preserves RED status on a receding limb when ``675.0 <= base_depth < 700.0`` and observed depth
    in the recent window (ordered chronologically from oldest to t0) reached ``>= 700.0 mm`` without
    dropping below ``675.0 mm`` since that peak.
    """
    b_d = float(base_depth)
    if b_d >= RED_DEPTH_MM:
        return "RED"
    if b_d >= RED_CLEAR_DEPTH_MM and recent_depths_mm is not None:
        arr = np.asarray(recent_depths_mm, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            red_indices = np.where(arr >= RED_DEPTH_MM)[0]
            if len(red_indices) > 0:
                last_red_idx = int(red_indices[-1])
                if bool(np.all(arr[last_red_idx:] >= RED_CLEAR_DEPTH_MM)):
                    return "RED"
    return "GREEN"


def evaluate_alert_hysteresis(
    base_depth: float,
    p50_series: list[float] | np.ndarray,
    p90_series: list[float] | np.ndarray,
    recent_depths_mm: pd.Series | np.ndarray | list[float] | None = None,
) -> list[str]:
    """Evaluate hourly caver safety alert statuses across a forecast trajectory with deadband hysteresis."""
    b_d = float(base_depth)
    p50_arr = np.asarray(p50_series, dtype=float)
    p90_arr = np.asarray(p90_series, dtype=float)
    n_steps = len(p50_arr)
    if n_steps == 0:
        return []

    prev_status = compute_initial_alert_status(b_d, recent_depths_mm=recent_depths_mm)
    statuses: list[str] = []

    for idx in range(n_steps):
        p50_v = float(p50_arr[idx])
        p90_v = float(p90_arr[idx])

        is_over_trigger = p50_v >= RED_TRIGGER_DEPTH_MM
        has_persist = False
        if is_over_trigger:
            if b_d >= AMBER_DEPTH_MM:
                has_persist = True
            else:
                prev_p50 = float(p50_arr[idx - 1]) if idx > 0 else b_d
                next_p50 = float(p50_arr[idx + 1]) if idx + 1 < n_steps else p50_v
                has_persist = (prev_p50 >= RED_DEPTH_MM) or (next_p50 >= RED_DEPTH_MM)

        is_recession_red = (prev_status == "RED") and (p50_v >= RED_CLEAR_DEPTH_MM)

        if (is_over_trigger and has_persist) or is_recession_red:
            status = "RED"
        elif (
            p90_v >= RED_DEPTH_MM
            or p50_v >= AMBER_DEPTH_MM
            or is_over_trigger
            or p50_v >= RED_CLEAR_DEPTH_MM
        ):
            status = "AMBER"
        else:
            status = "GREEN"

        statuses.append(status)
        prev_status = status

    return statuses


# Llanerch Yrfa (#1071) lags telemetry ~8h and correlates weakly with OFD1 (+2h CCF 0.365
# vs 0.546 for Dan Yr Ogof #1112 and 0.492 for Crai Reservoir #1051), so exclude it by default.
NRW_EXCLUDED_STATION_IDS: frozenset[int] = frozenset({1071})


def _nrw_station_list(payload: Any) -> list[dict[str, Any]]:
    """Normalize NRW StationData list, wrapper dict, or single Historical dict into a list."""
    if isinstance(payload, list):
        return [st for st in payload if isinstance(st, dict)]
    if isinstance(payload, dict):
        if "stations" in payload or "items" in payload:
            raw = payload.get("stations") or payload.get("items") or []
            return [st for st in raw if isinstance(st, dict)]
        if any(k in payload for k in ("parameterReadings", "readings", "parameters")):
            return [payload]
    return []


def extract_nrw_rain_targets(
    payload: Any,
    exclude_station_ids: set[int] | frozenset[int] | None = NRW_EXCLUDED_STATION_IDS,
) -> list[tuple[int, int, str]]:
    """Extract (location_id, parameter_id, title) rainfall targets from NRW byLocation JSON."""
    excluded = set(exclude_station_ids or ())
    targets: list[tuple[int, int, str]] = []
    for st in _nrw_station_list(payload):
        raw_loc = st.get("location") if st.get("location") is not None else st.get("stationId")
        try:
            loc_id = int(raw_loc)
        except (TypeError, ValueError):
            continue
        if loc_id in excluded:
            continue
        st_name = str(st.get("titleEn") or st.get("titleEN") or st.get("nameEN") or loc_id)
        params = st.get("parameters") if isinstance(st.get("parameters"), list) else [st]
        for param in params:
            if not isinstance(param, dict):
                continue
            p_name = str(param.get("paramNameEN") or "").lower()
            p_title = str(param.get("titleEn") or param.get("titleEN") or "").lower()
            if "rain" not in p_name and "rain" not in p_title and "rain" not in st_name.lower():
                continue
            raw_param = param.get("parameter")
            try:
                param_id = int(raw_param)
            except (TypeError, ValueError):
                continue
            targets.append((loc_id, param_id, st_name))
            break
    return targets


def _extract_nrw_parameter_hourly(
    param: dict[str, Any], min_samples_per_hour: int = 3
) -> pd.Series | None:
    """Aggregate a single NRW rainfall parameter dict into a 1h UTC rainfall Series."""
    readings = param.get("readings") or param.get("parameterReadings")
    if isinstance(readings, list) and readings:
        rows: list[tuple[pd.Timestamp, float]] = []
        for entry in readings:
            if not isinstance(entry, dict):
                continue
            raw_ts = entry.get("time") or entry.get("timestamp") or entry.get("dateTime")
            raw_val = entry.get("value") if "value" in entry else entry.get("val")
            if raw_ts is None or raw_val is None:
                continue
            ts = pd.to_datetime(raw_ts, utc=True, errors="coerce")
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                continue
            if pd.isna(ts) or not (0.0 <= val <= 100.0):
                continue
            rows.append((ts.floor("1h"), val))

        if not rows:
            return None
        s_raw = pd.DataFrame(rows, columns=["timestamp", "val"]).groupby("timestamp")["val"]
        counts = s_raw.count()
        totals = s_raw.sum()
        if int(counts.max()) > 1:
            valid = totals[counts >= min_samples_per_hour]
        else:
            valid = totals
        return valid.astype(float) if not valid.empty else None

    latest_val = param.get("latestValue")
    latest_time = param.get("latestTime")
    if latest_val is not None and latest_time is not None:
        ts = pd.to_datetime(latest_time, utc=True, errors="coerce")
        try:
            val = float(latest_val)
        except (TypeError, ValueError):
            return None
        if pd.notna(ts) and (0.0 <= val <= 100.0):
            return pd.Series([val], index=pd.DatetimeIndex([ts.floor("1h")], name="timestamp"), dtype=float)
    return None


def parse_nrw_rainfall_payload(
    payload: Any,
    min_samples_per_hour: int = 3,
    exclude_station_ids: set[int] | frozenset[int] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Parse NRW StationData byLocation or Historical JSON into an hourly UTC rainfall Series."""
    excluded = set(exclude_station_ids or ())
    stations = _nrw_station_list(payload)

    station_series: list[pd.Series] = []
    station_names: list[str] = []
    for st in stations:
        raw_loc = st.get("location") if st.get("location") is not None else st.get("stationId")
        if raw_loc is not None and excluded:
            try:
                if int(raw_loc) in excluded:
                    continue
            except (TypeError, ValueError):
                pass
        st_name = str(st.get("titleEn") or st.get("titleEN") or st.get("nameEN") or raw_loc or "NRW")
        params = st.get("parameters") if isinstance(st.get("parameters"), list) else [st]
        for param in params:
            if not isinstance(param, dict):
                continue
            p_name = str(param.get("paramNameEN") or "").lower()
            p_title = str(param.get("titleEn") or param.get("titleEN") or "").lower()
            if "rain" not in p_name and "rain" not in p_title and "rain" not in st_name.lower():
                continue
            extracted = _extract_nrw_parameter_hourly(param, min_samples_per_hour=min_samples_per_hour)
            if extracted is not None and not extracted.empty:
                station_series.append(extracted)
                station_names.append(st_name)
                break

    if not station_series:
        empty = pd.Series(dtype=float)
        empty.index = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return empty, {"stations_matched": 0, "station_names": []}

    combined = pd.concat(station_series, axis=1).mean(axis=1).sort_index()
    combined.index.name = "timestamp"
    return combined, {
        "stations_matched": len(station_names),
        "station_names": station_names,
    }
