from __future__ import annotations

import numpy as np
import pandas as pd

from app.hydrology import (
    CAUSAL_COLS,
    HORIZONS,
    MIN_PHYSICAL_DEPTH_MM,
    RED_CLEAR_DEPTH_MM,
    RED_DEPTH_MM,
    apply_hydrological_guardrails,
    attach_horizon_future_cols,
    combine_ensemble_quantile_deltas,
    compute_initial_alert_status,
    compute_nwp_ensemble_quantiles,
    engineer_features,
    enrich_nwp_ensemble_quantiles,
    evaluate_alert_hysteresis,
    reconstruct_depth_quantiles,
    roll_future_features_for_horizon,
    roll_future_moisture_series,
)


def test_feature_causality_under_future_perturbation(synthetic_hourly_df: pd.DataFrame) -> None:
    """Perturbing inputs strictly after t0 must never alter engineered features at t <= t0."""
    t0 = synthetic_hourly_df.index[500]
    df_base = engineer_features(synthetic_hourly_df)

    df_perturbed_in = synthetic_hourly_df.copy()
    future_mask = df_perturbed_in.index > t0
    df_perturbed_in.loc[future_mask, "depth_mm"] += 2000.0
    df_perturbed_in.loc[future_mask, "rain_mm"] += 120.0
    df_perturbed_in.loc[future_mask, "swcc_rain_mm"] += 120.0
    df_perturbed_in.loc[future_mask, "stump_rain_mm"] += 120.0
    df_perturbed_in.loc[future_mask, "era5_soil_moisture"] = 0.99
    df_perturbed_in.loc[future_mask, "wind_kmh"] = 95.0

    df_pert = engineer_features(df_perturbed_in)

    past_base = df_base.loc[:t0, CAUSAL_COLS]
    past_pert = df_pert.loc[:t0, CAUSAL_COLS]
    for col in CAUSAL_COLS:
        np.testing.assert_allclose(
            past_base[col].values,
            past_pert[col].values,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Causality violation in column {col} at t <= t0",
        )


def test_stump_glitch_filter_bounds(synthetic_hourly_df: pd.DataFrame) -> None:
    """Single-gauge Stump spikes (>=15 mm with SWCC <5 and reference <5) are masked; corroborated rain is kept."""
    df_feat = engineer_features(synthetic_hourly_df)

    # Hours 200..203 had stump_rain_mm = 96.6 while swcc = 0.0 and rain = 0.0
    glitch_slice = df_feat.iloc[200:204]
    assert bool(glitch_slice["stump_glitch_flag"].all()) is True
    assert bool(glitch_slice["stump_rain_clean"].isna().all()) is True
    np.testing.assert_allclose(glitch_slice["fused_rain_mm"].values, 0.0)

    # Add a corroborated heavy storm hour (>=15 mm on Stump and >=5 mm on SWCC)
    df_corrob = synthetic_hourly_df.copy()
    df_corrob.iloc[250, df_corrob.columns.get_loc("stump_rain_mm")] = 18.0
    df_corrob.iloc[250, df_corrob.columns.get_loc("swcc_rain_mm")] = 12.0
    df_corrob.iloc[250, df_corrob.columns.get_loc("rain_mm")] = 10.0
    df_corrob_feat = engineer_features(df_corrob)

    row_250 = df_corrob_feat.iloc[250]
    assert bool(row_250["stump_glitch_flag"]) is False
    assert float(row_250["stump_rain_clean"]) == 18.0
    assert float(row_250["club_rain_mean"]) == 15.0


def test_asymmetric_fut_ref_tank_24h_and_rolling_equivalence(
    synthetic_hourly_df: pd.DataFrame,
) -> None:
    """Verify attach_horizon_future_cols preserves fut_ref_tank_24h = tank_24h.shift(-h) and matches roll_future_features_for_horizon."""
    df_in = synthetic_hourly_df.copy()
    # Make fused_rain_mm equal rain_mm across future window so tank_24h and roll_future_features_for_horizon see identical rain
    df_in["swcc_rain_mm"] = df_in["rain_mm"]
    df_in["stump_rain_mm"] = df_in["rain_mm"]
    df_feat = engineer_features(df_in)

    issue_idx = df_feat.index[390:405]
    fut_rain_mat = np.column_stack(
        [df_feat["rain_mm"].shift(-step).loc[issue_idx].values for step in range(1, 49)]
    )

    for h in HORIZONS:
        attached = attach_horizon_future_cols(df_feat, h)
        # Verify asymmetric definition: 6h and 96h shift ref_tank, 24h shifts tank_24h
        np.testing.assert_allclose(
            attached.loc[issue_idx, "fut_ref_tank_6h"].values,
            df_feat["ref_tank_6h"].shift(-h).loc[issue_idx].values,
        )
        np.testing.assert_allclose(
            attached.loc[issue_idx, "fut_ref_tank_24h"].values,
            df_feat["tank_24h"].shift(-h).loc[issue_idx].values,
        )
        np.testing.assert_allclose(
            attached.loc[issue_idx, "fut_ref_tank_96h"].values,
            df_feat["ref_tank_96h"].shift(-h).loc[issue_idx].values,
        )

        rolled = roll_future_features_for_horizon(df_feat, issue_idx, fut_rain_mat, h)
        for col in ["fut_ref_tank_6h", "fut_ref_tank_24h", "fut_ref_tank_96h", "fut_rain_sum_6h_at_target", "fut_rain_sum_12h_at_target"]:
            np.testing.assert_allclose(
                rolled[col].values,
                attached.loc[issue_idx, col].values,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Mismatch in {col} at horizon +{h}h",
            )
        for k in range(min(h, 6)):
            col_k = f"fut_rain_minus_{k}h_from_target"
            np.testing.assert_allclose(
                rolled[col_k].values,
                attached.loc[issue_idx, col_k].values,
                rtol=1e-10,
                atol=1e-10,
            )


def test_roll_future_moisture_series_uses_full_history_bounds(
    synthetic_hourly_df: pd.DataFrame,
) -> None:
    """Verify roll_future_moisture_series normalizes against full-history bounds stored on now_row."""
    df_feat = engineer_features(synthetic_hourly_df)
    now_ts = df_feat.index[-1]
    now_row = df_feat.loc[now_ts]
    fut_idx = pd.date_range(now_ts + pd.Timedelta(hours=1), periods=48, freq="1h", tz="UTC")
    fut_wx = pd.DataFrame(
        {
            "rain_p50_mm": np.full(48, 2.0),
            "et0_mm": np.full(48, 0.05),
            "era5_soil_moisture": np.full(48, 0.40),
        },
        index=fut_idx,
    )
    moist_s = roll_future_moisture_series(now_row, fut_wx)
    assert len(moist_s) == 48
    assert bool((moist_s >= 0.05).all() and (moist_s <= 1.25).all())
    assert float(moist_s.iloc[-1]) > float(moist_s.iloc[0])


def test_nwp_wet_quorum_evaluated_before_nan_imputation() -> None:
    """Missing NWP member columns (NaN) must not count toward the 2-model >=0.6 mm/h wet quorum."""
    # 6 models across 48 hours:
    # Model 0 (best_match): 1.2 mm/h
    # Model 1..4: 0.0 mm/h
    # Model 5: NaN (missing model column)
    raw_2d = np.zeros((6, 48), dtype=float)
    raw_2d[0, :] = 1.2
    raw_2d[5, :] = np.nan
    det_2d = np.full(48, 1.2, dtype=float)

    # With only 1 non-NaN model >= 0.6, wet_members_count == 1 (< 2), so outlier damping must reduce raw_ens_max
    p10_1m, p50_1m, p90_1m, _ = compute_nwp_ensemble_quantiles(raw_2d, det_2d)

    # Now set Model 1 to 1.2 mm/h as well -> 2 actual models >= 0.6 (quorum met)
    raw_2d_quorum = raw_2d.copy()
    raw_2d_quorum[1, :] = 1.2
    _, _, p90_2m, _ = compute_nwp_ensemble_quantiles(raw_2d_quorum, det_2d)

    assert bool(np.all(p90_1m < p90_2m)), (
        "NaN member filled with det_rain=1.2 falsely triggered the 2-model wet quorum!"
    )

    # Verify 3D (M, N, T) consistency with 2D (M, T)
    raw_3d = np.stack([raw_2d, raw_2d_quorum], axis=1)  # shape (6, 2, 48)
    det_3d = np.stack([det_2d, det_2d], axis=0)  # shape (2, 48)
    p10_3d, p50_3d, p90_3d, _ = compute_nwp_ensemble_quantiles(raw_3d, det_3d)
    np.testing.assert_allclose(p10_3d[0], p10_1m)
    np.testing.assert_allclose(p50_3d[0], p50_1m)
    np.testing.assert_allclose(p90_3d[0], p90_1m)
    np.testing.assert_allclose(p90_3d[1], p90_2m)


def test_enrich_nwp_ensemble_quantiles_modes() -> None:
    """Verify deterministic and ensemble enrichment produce valid metadata and monotonic rain quantiles."""
    idx = pd.date_range("2026-11-01 00:00:00", periods=48, freq="1h", tz="UTC")
    fut_wx = pd.DataFrame(
        {
            "rain_mm": np.full(48, 1.5),
            "nwp_ukmo_seamless": np.full(48, 2.0),
            "nwp_ecmwf_ifs025": np.full(48, 1.0),
            "nwp_icon_seamless": np.full(48, 0.2),
        },
        index=idx,
    )
    df_ens, meta_ens = enrich_nwp_ensemble_quantiles(fut_wx, {"press_hpa": 990.0}, use_ensemble=True)
    assert meta_ens["models_count"] == 3
    assert bool((df_ens["rain_p10_mm"] <= df_ens["rain_p50_mm"]).all())
    assert bool((df_ens["rain_p50_mm"] <= df_ens["rain_p90_mm"]).all())

    df_det, meta_det = enrich_nwp_ensemble_quantiles(fut_wx, {"press_hpa": 990.0}, use_ensemble=False)
    assert meta_det["models_count"] == 1
    np.testing.assert_allclose(df_det["rain_p10_mm"].values, df_det["rain_p90_mm"].values)


def test_guardrails_reference_frame_and_monotonic_quantiles() -> None:
    """Verify d_p10 reference-frame fix preserves lower spread on receding and clamped hours."""
    # Case 1: Receding limb from 600 mm (d_p50 = -80 mm, total_low_spread = 25 mm, total_high_spread = 30 mm)
    base_d = 600.0
    d_p10_rel, d_p50_rel, d_p90_rel, d_hydro_hi = combine_ensemble_quantile_deltas(
        d_p50=-80.0,
        d_p10_on_p50=-100.0,
        d_p50_on_p10=-95.0,
        d_p90_on_p50=-60.0,
        d_p90_on_p90=-50.0,
        d_p50_on_p90=-58.0,
    )
    assert d_p10_rel < d_p50_rel < d_p90_rel

    d10_c, d50_c, d90_c = apply_hydrological_guardrails(
        base_depth=base_d,
        ddepth_1h=-8.0,
        fused_rain_3h=0.0,
        moisture_index=0.6,
        tank_168h=0.6,
        horizon_h=12,
        fut_rain_sum=0.0,
        d_p10=d_p10_rel,
        d_p50=d_p50_rel,
        d_p90=d_p90_rel,
    )
    p10, p50, p90, p90_hydro = reconstruct_depth_quantiles(
        base_d, d10_c, d50_c, d90_c, delta_hydro_high=d_hydro_hi
    )
    assert MIN_PHYSICAL_DEPTH_MM <= p10 < p50 <= p90_hydro <= p90
    # Lower spread (p50 - p10) must NOT collapse to 0 on a receding limb!
    assert np.isclose(p50 - p10, d_p50_rel - d_p10_rel)

    # Case 2: Unphysical dry-bog spike when base_depth < 620 mm and ddepth_1h < 15 mm
    # Model predicts +200 mm rise with 0 mm rain and dry bog; guardrail clamps p50 to +25 mm
    d10_spike, d50_spike, d90_spike = apply_hydrological_guardrails(
        base_depth=520.0,
        ddepth_1h=0.0,
        fused_rain_3h=0.0,
        moisture_index=0.3,
        tank_168h=0.2,
        horizon_h=6,
        fut_rain_sum=0.0,
        d_p10=170.0,  # 200 - 30 spread
        d_p50=200.0,
        d_p90=250.0,
    )
    p10_s, p50_s, p90_s = reconstruct_depth_quantiles(520.0, d10_spike, d50_spike, d90_spike)
    assert p50_s == 545.0  # 520 + 25 mm max physical dry rise
    assert p10_s == 515.0  # preserves the 30 mm lower spread below clamped p50!
    assert MIN_PHYSICAL_DEPTH_MM <= p10_s <= p50_s <= p90_s < RED_DEPTH_MM


def test_alert_hysteresis_rising_falling_and_t0_recession() -> None:
    """Verify rising debounce (705 mm), falling deadband (675 mm), and t0 receding-limb initialization."""
    # 1. Single-hour edge touch at 706 mm from quiescent base (550 mm) without persistence stays AMBER
    p50_touch = [600.0, 706.0, 650.0, 580.0]
    p90_touch = [630.0, 720.0, 680.0, 600.0]
    statuses_touch = evaluate_alert_hysteresis(550.0, p50_touch, p90_touch)
    assert statuses_touch[1] == "AMBER"

    # 2. Multi-hour flood pulse (>= 705 mm with neighbor >= 700 mm) triggers RED and holds down to 675 mm
    p50_flood = [680.0, 706.0, 725.0, 695.0, 675.0, 674.0, 610.0]
    p90_flood = [710.0, 740.0, 760.0, 720.0, 690.0, 685.0, 615.0]
    statuses_flood = evaluate_alert_hysteresis(600.0, p50_flood, p90_flood)
    assert statuses_flood == ["AMBER", "RED", "RED", "RED", "RED", "AMBER", "GREEN"]

    # 3. t0 receding-limb initialization: base_depth is 685 mm (between 675 and 700 mm) after a 720 mm peak 2h ago
    recent_receding = [650.0, 710.0, 720.0, 698.0, 685.0]
    assert compute_initial_alert_status(685.0, recent_depths_mm=recent_receding) == "RED"
    statuses_receding = evaluate_alert_hysteresis(
        685.0,
        p50_series=[680.0, 675.0, 670.0, 600.0],
        p90_series=[690.0, 685.0, 680.0, 610.0],
        recent_depths_mm=recent_receding,
    )
    assert statuses_receding == ["RED", "RED", "AMBER", "GREEN"]

    # Contrast with rising limb at 685 mm that has NOT yet reached 700 mm
    recent_rising = [560.0, 610.0, 645.0, 670.0, RED_CLEAR_DEPTH_MM + 10.0]
    assert compute_initial_alert_status(685.0, recent_depths_mm=recent_rising) == "GREEN"


def test_stuck_zero_rain_gauge_quarantine_and_fallback(synthetic_hourly_df: pd.DataFrame) -> None:
    """Stuck-zero gauge for >=6h while stream rises and Open-Meteo shows rain triggers fallback to rain_mm."""
    df_in = synthetic_hourly_df.copy()
    df_in["swcc_rain_mm"] = np.nan
    df_in["stump_rain_mm"] = 0.0
    # Inject a 20 mm streamway rise + Open-Meteo rain around index 300
    df_in.iloc[295:302, df_in.columns.get_loc("rain_mm")] = 1.5
    df_in.iloc[300:303, df_in.columns.get_loc("depth_mm")] += 25.0

    df_feat = engineer_features(df_in)
    assert df_feat["stump_status"].iloc[300] == "stuck_zero"
    assert df_feat["rain_source"].iloc[300] == "open_meteo"
    assert df_feat["fused_rain_mm"].iloc[300] == 1.5


def test_depth_step_drop_clamps_momentum_without_masking_depth(
    synthetic_hourly_df: pd.DataFrame,
) -> None:
    """Non-flood sudden depth drop clamps momentum derivatives while keeping base depth_mm intact."""
    df_in = synthetic_hourly_df.copy()
    df_in.iloc[350:355, df_in.columns.get_loc("depth_mm")] = 680.0
    df_in.iloc[355:360, df_in.columns.get_loc("depth_mm")] = 584.0

    df_feat = engineer_features(df_in)
    assert df_feat["depth_mm"].iloc[355] == 584.0
    assert bool(df_feat["depth_step_drop_flag"].iloc[355]) is True
    assert df_feat["ddepth_1h"].iloc[355] >= -10.0
    assert df_feat["accel_1h"].iloc[355] >= -5.0


def test_guardrails_decouple_p50_and_p90_for_short_horizons() -> None:
    """At h<=6h, unconfirmed NWP rain can raise P90 AMBER watch without unlocking P50 RED rise cap."""
    p10, p50, p90, tel = apply_hydrological_guardrails(
        base_depth=575.0,
        ddepth_1h=0.0,
        fused_rain_3h=12.0,
        moisture_index=0.8,
        tank_168h=0.6,
        horizon_h=6,
        fut_rain_sum=15.0,
        d_p10=10.0,
        d_p50=150.0,
        d_p90=220.0,
        obs_rain_confirmed=False,
        return_telemetry=True,
    )
    assert p50 == 25.0
    assert p90 > 25.0
    assert p10 <= p50 <= p90
    assert tel["guardrail_clamped"] is True
    assert tel["p50_clamp_shift_mm"] == 125.0


def test_nrw_fallback_chain_and_payload_parser() -> None:
    """Verify NRW 15-min/hourly parsing, completeness threshold, and club -> nrw -> open_meteo chain."""
    from app.hydrology import parse_nrw_rainfall_payload

    payload = [
        {
            "location": 9001,
            "titleEn": "Tawe Upper Rain",
            "parameters": [
                {
                    "paramNameEN": "Rainfall",
                    "readings": [
                        # Complete 4x15-min hour (sum = 3.0 mm)
                        {"time": "2026-11-10T10:00:00Z", "value": 0.5},
                        {"time": "2026-11-10T10:15:00Z", "value": 1.0},
                        {"time": "2026-11-10T10:30:00Z", "value": 1.0},
                        {"time": "2026-11-10T10:45:00Z", "value": 0.5},
                        # Incomplete sub-hourly bucket (2 samples < 3 -> rejected)
                        {"time": "2026-11-10T11:00:00Z", "value": 2.0},
                        {"time": "2026-11-10T11:15:00Z", "value": 2.0},
                    ],
                }
            ],
        },
        {
            "location": 9002,
            "titleEn": "Cray Headwater",
            "parameters": [
                {
                    "paramNameEN": "Rainfall",
                    "latestValue": 4.2,
                    "latestTime": "2026-11-10T12:15:00Z",
                }
            ],
        },
    ]

    nrw_series, meta = parse_nrw_rainfall_payload(payload, min_samples_per_hour=3)
    assert meta["stations_matched"] == 2
    assert pd.Timestamp("2026-11-10T10:00:00Z") in nrw_series.index
    assert nrw_series.loc[pd.Timestamp("2026-11-10T10:00:00Z")] == 3.0
    assert pd.Timestamp("2026-11-10T11:00:00Z") not in nrw_series.index
    assert nrw_series.loc[pd.Timestamp("2026-11-10T12:00:00Z")] == 4.2

    idx = pd.date_range("2026-11-01", periods=400, freq="1h", tz="UTC")
    df_in = pd.DataFrame(
        {
            "depth_mm": 550.0,
            "stump_rain_mm": np.nan,
            "swcc_rain_mm": np.nan,
            "nrw_rain_mm": np.nan,
            "rain_mm": 0.4,
            "wind_kmh": 10.0,
            "et0_mm": 0.05,
            "era5_soil_moisture": 0.35,
            "era5_soil_deep": 0.35,
            "temp_c": 9.0,
            "era5_press_hpa": 1005.0,
            "swcc_press_hpa": np.nan,
            "stump_press_hpa": np.nan,
        },
        index=idx,
    )
    df_in.iloc[360, df_in.columns.get_loc("nrw_rain_mm")] = 2.8

    df_feat = engineer_features(df_in)
    assert df_feat["rain_source"].iloc[360] == "nrw"
    assert df_feat["nrw_status"].iloc[360] == "ok"
    assert df_feat["fused_rain_mm"].iloc[360] == 2.8
    assert df_feat["rain_source"].iloc[361] == "open_meteo"
    assert df_feat["fused_rain_mm"].iloc[361] == 0.4


def test_nrw_historical_payload_and_target_filtering() -> None:
    """Verify NRW Historical parameterReadings parsing and exclusion of lagging station 1071."""
    from app.hydrology import (
        NRW_EXCLUDED_STATION_IDS,
        extract_nrw_rain_targets,
        parse_nrw_rainfall_payload,
    )

    by_loc = [
        {
            "location": 1051,
            "titleEn": "Crai Reservoir raingauge",
            "parameters": [{"parameter": 10146, "paramNameEN": "Rainfall"}],
        },
        {
            "location": 1071,
            "titleEn": "Llanerch Yrfa raingauge",
            "parameters": [{"parameter": 10166, "paramNameEN": "Rainfall"}],
        },
        {
            "location": 1112,
            "titleEn": "Dan Yr Ogof raingauge",
            "parameters": [{"parameter": 10207, "paramNameEN": "Rainfall"}],
        },
        {
            "location": 4091,
            "titleEn": "Tawe at Craig y Nos",
            "parameters": [{"parameter": 152, "paramNameEN": "River Level"}],
        },
    ]
    targets = extract_nrw_rain_targets(by_loc, exclude_station_ids=NRW_EXCLUDED_STATION_IDS)
    assert [loc for loc, _, _ in targets] == [1051, 1112]

    hist_payloads = [
        {
            "location": 1051,
            "titleEN": "Crai Reservoir raingauge",
            "parameter": 10146,
            "paramNameEN": "Rainfall",
            "parameterReadings": [
                {"time": "2026-11-10T18:00:00Z", "value": 0.4},
                {"time": "2026-11-10T18:15:00Z", "value": 0.4},
                {"time": "2026-11-10T18:30:00Z", "value": 0.4},
                {"time": "2026-11-10T18:45:00Z", "value": 0.4},
            ],
        },
        {
            "location": 1112,
            "titleEN": "Dan Yr Ogof raingauge",
            "parameter": 10207,
            "paramNameEN": "Rainfall",
            "parameterReadings": [
                {"time": "2026-11-10T18:00:00Z", "value": 0.2},
                {"time": "2026-11-10T18:15:00Z", "value": 0.2},
                {"time": "2026-11-10T18:30:00Z", "value": 0.2},
                {"time": "2026-11-10T18:45:00Z", "value": 0.4},
            ],
        },
    ]
    ser, meta = parse_nrw_rainfall_payload(hist_payloads, exclude_station_ids=NRW_EXCLUDED_STATION_IDS)
    assert meta["stations_matched"] == 2
    assert ser.loc[pd.Timestamp("2026-11-10T18:00:00Z")] == 1.3
