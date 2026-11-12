"""Fast, exact out-of-sample 2026 backtest of the 6-model NWP ensemble vs. deterministic NWP.

Evaluates every issue hour in the 2026 unseen test set (models trained strictly on targets < 2026-01-01)
across three issue-time weather scenarios:

* ``deterministic`` – Open-Meteo ``best_match`` rain as forecast at issue time, hydrological P10–P90 band only.
* ``ensemble``      – 6-model NWP ensemble (UKMO, ECMWF, ICON, AROME, GFS, Blend) as forecast at issue time,
  volume-conserved P10/P50/P90 rain trajectories + quadrature-combined P10–P90 band.
* ``perfect_wx``    – Observed/reanalysis rain (reference upper bound).

Writes ``data/nwp_ensemble_backtest.parquet`` and ``data/nwp_ensemble_backtest_summary.csv``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine import (  # noqa: E402
    CAUSAL_COLS,
    RED_DEPTH_MM,
    TRAIN_SPLIT_UTC,
    OFD1ForecastEngine,
    nwp_cols_for_horizon,
)

MODELS = [
    "best_match",
    "ukmo_seamless",
    "ecmwf_ifs025",
    "icon_seamless",
    "gfs_seamless",
    "meteofrance_seamless",
]
EVAL_HORIZONS = [6, 12, 18, 24, 36, 48]


def _build_future_rain_trajectories(
    arch: pd.DataFrame, valid_obs: pd.DataFrame, issue_idx: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (N, 48) matrices of future hourly rain for:
    (det_rain, ens_p10, ens_p50, ens_p90, perf_rain, member_spread_48h)
    """
    n_issues = len(issue_idx)
    # Build 2D index of timestamps: t0 + 1h .. t0 + 48h
    full_idx = pd.date_range(
        issue_idx.min() + pd.Timedelta(hours=1),
        issue_idx.max() + pd.Timedelta(hours=48),
        freq="1h",
        tz="UTC",
    )
    arch_re = arch.reindex(full_idx)
    obs_rain_s = valid_obs["rain_mm"].reindex(full_idx).fillna(0.0).values

    # Map each issue_idx to its starting integer offset in full_idx
    base_offsets = ((issue_idx - full_idx[0]) // pd.Timedelta(hours=1)).values.astype(int)
    # (N, 48) index grid into full_idx
    grid_idx = base_offsets[:, None] + np.arange(0, 48, dtype=int)[None, :]

    # For each model slug, construct (N, 48) matrix using d0 for lead 1..6, d1 for 7..30, d2 for 31..48
    lead_h = np.arange(1, 49)
    m_d0 = lead_h <= 6
    m_d1 = (lead_h > 6) & (lead_h <= 30)
    m_d2 = lead_h > 30

    member_mats = []
    for slug in MODELS:
        c0 = arch_re[f"{slug}_d0"].values if f"{slug}_d0" in arch_re else np.full(len(full_idx), np.nan)
        c1 = arch_re[f"{slug}_d1"].values if f"{slug}_d1" in arch_re else np.full(len(full_idx), np.nan)
        c2 = arch_re[f"{slug}_d2"].values if f"{slug}_d2" in arch_re else np.full(len(full_idx), np.nan)
        mat = np.empty((n_issues, 48), dtype=float)
        mat[:, m_d0] = c0[grid_idx[:, m_d0]]
        mat[:, m_d1] = c1[grid_idx[:, m_d1]]
        mat[:, m_d2] = c2[grid_idx[:, m_d2]]
        member_mats.append(mat)

    # Stack shape: (M, N, 48)
    stack = np.stack(member_mats, axis=0)
    det_rain = np.where(np.isnan(stack[0]), np.nanmedian(stack, axis=0), stack[0])
    det_rain = np.nan_to_num(det_rain, nan=0.0)

    # Fill missing member values with det_rain just as _enrich_nwp_ensemble_quantiles does
    stack_filled = np.where(np.isnan(stack), det_rain[None, :, :], stack)

    ens_min = np.percentile(stack_filled, 10, axis=0)
    ens_med = np.percentile(stack_filled, 50, axis=0)
    raw_ens_max = np.percentile(stack_filled, 90, axis=0)

    # 2-model consensus quorum: require >= 2 members predicting rain >= 0.6 mm/h
    wet_members_count = np.sum(stack_filled >= 0.6, axis=0)
    ens_max = np.where(
        wet_members_count >= 2,
        raw_ens_max,
        np.minimum(raw_ens_max, 0.5 * (ens_med + raw_ens_max)),
    )

    rain_p50 = np.round(0.5 * det_rain + 0.5 * ens_med, 2)
    rain_p10 = np.round(np.minimum(rain_p50, ens_min), 2)

    member_totals_48h = np.sum(stack_filled, axis=2)  # (M, N)
    high_vol_count = np.sum(member_totals_48h >= 10.0, axis=0)  # (N,)
    target_p90_raw = np.percentile(member_totals_48h, 90, axis=0)  # (N,)
    target_p75_raw = np.percentile(member_totals_48h, 75, axis=0)  # (N,)
    target_p90_total = np.where(high_vol_count >= 2, target_p90_raw, target_p75_raw)

    raw_p90 = np.maximum(rain_p50, ens_max)
    raw_p90_sum = np.sum(raw_p90, axis=1)  # (N,)

    vol_scale = np.where(
        (raw_p90_sum > 1e-3) & (target_p90_total > 0.0),
        np.clip(target_p90_total / np.maximum(raw_p90_sum, 1e-6), 0.55, 1.15),
        1.0,
    )
    rain_p90 = np.round(np.maximum(rain_p50, raw_p90 * vol_scale[:, None]), 2)
    perf_rain = obs_rain_s[grid_idx]
    spread_48h = np.max(member_totals_48h, axis=0) - np.min(member_totals_48h, axis=0)

    return det_rain, rain_p10, rain_p50, rain_p90, perf_rain, spread_48h


def _roll_future_features_for_horizon(
    valid_obs: pd.DataFrame,
    issue_idx: pd.DatetimeIndex,
    fut_rain_mat: np.ndarray,
    h: int,
) -> pd.DataFrame:
    """Compute exact CAUSAL_COLS + nwp_cols_for_horizon(h) at issue_idx for a given (N, 48) future rain matrix."""
    sub = valid_obs.loc[issue_idx, CAUSAL_COLS].copy()
    # We need last 11 hours of observed rain before t0 for 6h/12h rolling sums when h < 12
    full_idx = valid_obs.index
    base_pos = full_idx.get_indexer(issue_idx)

    # Gather past 11h of ref_rain (t0-10h .. t0) and eff_rain (t0-5h .. t0)
    rain_obs_all = valid_obs["rain_mm"].fillna(valid_obs["fused_rain_mm"]).fillna(0.0).values
    eff_obs_all = valid_obs["eff_rain_mm"].fillna(0.0).values
    et0_t0 = valid_obs.loc[issue_idx, "et0_mm"].fillna(0.05).values
    wind_t0 = valid_obs.loc[issue_idx, "wind_kmh"].fillna(15.0).values
    wind_factor = 1.0 + 0.004 * np.clip(wind_t0 - 15.0, 0.0, None)
    moist_t0 = valid_obs.loc[issue_idx, "moisture_index"].fillna(0.5).values

    # Roll EWM tanks from t0 to t0 + h
    ref_tank_6 = valid_obs.loc[issue_idx, "ref_tank_6h"].values.copy()
    tank_24 = valid_obs.loc[issue_idx, "tank_24h"].values.copy()
    ref_tank_96 = valid_obs.loc[issue_idx, "ref_tank_96h"].values.copy()

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

    # Build combined rain window [t0-11h .. t0, t0+1h .. t0+h]
    past_12 = np.column_stack([rain_obs_all[ np.maximum(0, base_pos - k) ] for k in range(11, -1, -1)])
    comb_rain = np.hstack([past_12, fut_rain_mat[:, :h]])  # index 11 is t0, index 11+h is t0+h
    tgt_col = 11 + h
    sub["fut_rain_sum_6h_at_target"] = np.sum(comb_rain[:, tgt_col - 5 : tgt_col + 1], axis=1)
    sub["fut_rain_sum_12h_at_target"] = np.sum(comb_rain[:, tgt_col - 11 : tgt_col + 1], axis=1)

    fut_eff_mat = (fut_rain_mat[:, :h] * wind_factor[:, None]) * (moist_t0[:, None] ** 1.5)
    past_6_eff = np.column_stack([eff_obs_all[ np.maximum(0, base_pos - k) ] for k in range(5, -1, -1)])
    comb_eff = np.hstack([past_6_eff, fut_eff_mat])  # index 5 is t0, index 5+h is t0+h
    eff_tgt_col = 5 + h
    sub["fut_eff_rain_sum_6h_at_target"] = np.sum(comb_eff[:, eff_tgt_col - 5 : eff_tgt_col + 1], axis=1)

    for k in range(0, min(h, 6)):
        sub[f"fut_rain_minus_{k}h_from_target"] = fut_rain_mat[:, h - 1 - k]

    return sub[CAUSAL_COLS + nwp_cols_for_horizon(h)]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["actual_depth_mm"]).copy()
    y, lo, md, hi = df["actual_depth_mm"], df["p10_mm"], df["p50_mm"], df["p90_mm"]
    alpha = 0.2
    df["covered"] = (y >= lo) & (y <= hi)
    df["width"] = hi - lo
    df["abs_err"] = (md - y).abs()
    df["interval_score"] = (
        (hi - lo) + (2 / alpha) * (lo - y).clip(lower=0) + (2 / alpha) * (y - hi).clip(lower=0)
    )
    df["obs_red"] = y >= RED_DEPTH_MM
    df["p90_red"] = hi >= RED_DEPTH_MM
    df["p50_red"] = md >= RED_DEPTH_MM

    def _agg(g: pd.DataFrame) -> pd.Series:
        obs = g["obs_red"]
        wet = g[g["member_rain_48h_spread_mm"] >= 2.0]
        out = {
            "n": len(g),
            "coverage_pct": 100 * g["covered"].mean(),
            "wet_coverage_pct": 100 * wet["covered"].mean() if len(wet) else np.nan,
            "mean_width_mm": g["width"].mean(),
            "wet_width_mm": wet["width"].mean() if len(wet) else np.nan,
            "mae_p50_mm": g["abs_err"].mean(),
            "interval_score": g["interval_score"].mean(),
            "obs_red_hours": int(obs.sum()),
        }
        for q in ("p90", "p50"):
            pred = g[f"{q}_red"]
            hits = int((pred & obs).sum())
            fa = int((pred & ~obs).sum())
            miss = int((~pred & obs).sum())
            out[f"{q}_red_pod_pct"] = 100 * hits / max(1, hits + miss)
            out[f"{q}_red_far_pct"] = 100 * fa / max(1, hits + fa)
            out[f"{q}_red_false_alarms"] = fa
            out[f"{q}_red_misses"] = miss
        return pd.Series(out)

    summ = df.groupby(["horizon_h", "scenario"]).apply(_agg, include_groups=False).reset_index()
    return summ.round(2)


def main() -> None:
    t0 = time.time()
    eng = OFD1ForecastEngine(data_dir="data")
    eng.df_hourly = pd.read_parquet(eng.aligned_1h_path).sort_index()
    eng.models = joblib.load(eng.models_path)
    df_feat = eng._engineer_features(eng.df_hourly)
    valid_obs = df_feat.dropna(subset=["depth_mm"] + CAUSAL_COLS).sort_index()
    arch = pd.read_parquet(Path("data/ofd1_nwp_archive.parquet")).sort_index()

    end_ts = min(valid_obs.index.max(), arch.index.max()) - pd.Timedelta(hours=48)
    issue_idx = valid_obs.loc[TRAIN_SPLIT_UTC:end_ts].index
    print(f"Vectorized backtest over {len(issue_idx)} hourly 2026 issue times ({issue_idx[0]} → {issue_idx[-1]})")

    det_r, p10_r, p50_r, p90_r, perf_r, spread_48h = _build_future_rain_trajectories(
        arch, valid_obs, issue_idx
    )
    base_d = valid_obs.loc[issue_idx, "depth_mm"].values

    chunks = []
    for h in EVAL_HORIZONS:
        X_det = _roll_future_features_for_horizon(valid_obs, issue_idx, det_r, h)
        X_p10 = _roll_future_features_for_horizon(valid_obs, issue_idx, p10_r, h)
        X_p50 = _roll_future_features_for_horizon(valid_obs, issue_idx, p50_r, h)
        X_p90 = _roll_future_features_for_horizon(valid_obs, issue_idx, p90_r, h)
        X_perf = _roll_future_features_for_horizon(valid_obs, issue_idx, perf_r, h)

        m = eng.models[h]
        target_idx = issue_idx + pd.Timedelta(hours=h)
        act_d = valid_obs["depth_mm"].reindex(target_idx).values

        # 1. Deterministic NWP (hydrological P10-P90 only)
        d50_det = m["p50"].predict(X_det)
        d10_det = m["p10"].predict(X_det)
        d90_det = m["p90"].predict(X_det)
        p50_det = np.maximum(350.0, base_d + d50_det)
        p10_det = np.maximum(350.0, np.minimum(p50_det, base_d + d10_det))
        p90_det = np.maximum(p50_det, base_d + d90_det)

        # 2. 6-Model NWP Ensemble (quadrature combination matching _forecast_anchors)
        d_p50 = m["p50"].predict(X_p50)
        d_p10_on_p10 = m["p10"].predict(X_p10)
        d_p50_on_p10 = m["p50"].predict(X_p10)
        delta_hydro_low = np.maximum(0.0, d_p50 - m["p10"].predict(X_p50))
        delta_nwp_low = np.maximum(0.0, d_p50 - d_p50_on_p10)
        total_low = np.sqrt(delta_hydro_low**2 + delta_nwp_low**2)

        d_p90_on_p50 = m["p90"].predict(X_p50)
        d_p90_on_p90 = m["p90"].predict(X_p90)
        d_p50_on_p90 = m["p50"].predict(X_p90)
        delta_hydro_high = np.maximum(0.0, d_p90_on_p50 - d_p50)
        delta_nwp_high = np.maximum(
            0.0, np.maximum(d_p50_on_p90 - d_p50, 0.35 * (d_p90_on_p90 - d_p90_on_p50))
        )
        total_high = np.sqrt(delta_hydro_high**2 + delta_nwp_high**2)

        fused_3h = (
            valid_obs.loc[issue_idx, "fused_rain_sum_3h"].values
            if "fused_rain_sum_3h" in valid_obs.columns
            else np.zeros(len(issue_idx))
        )
        fut_sum = (
            X_p50["fut_rain_sum_6h_at_target"].values
            if "fut_rain_sum_6h_at_target" in X_p50.columns
            else np.zeros(len(issue_idx))
        )
        moist_idx = (
            valid_obs.loc[issue_idx, "moisture_index"].values
            if "moisture_index" in valid_obs.columns
            else np.full(len(issue_idx), 0.5)
        )
        t_168 = (
            valid_obs.loc[issue_idx, "tank_168h"].values
            if "tank_168h" in valid_obs.columns
            else np.full(len(issue_idx), 0.5)
        )
        dd_1h = (
            valid_obs.loc[issue_idx, "ddepth_1h"].values
            if "ddepth_1h" in valid_obs.columns
            else np.zeros(len(issue_idx))
        )

        d10_c, d50_c, d90_c = eng._apply_hydrological_guardrails(
            base_depth=base_d,
            ddepth_1h=dd_1h,
            fused_rain_3h=fused_3h,
            moisture_index=moist_idx,
            tank_168h=t_168,
            horizon_h=h,
            fut_rain_sum=fut_sum,
            d_p10=-total_low,
            d_p50=d_p50,
            d_p90=d_p50 + total_high,
        )

        p50_ens = np.maximum(350.0, base_d + d50_c)
        p10_ens = np.maximum(350.0, base_d + np.minimum(d10_c, d50_c))
        p90_ens = np.maximum(p50_ens, base_d + d90_c)

        # 3. Perfect Wx reference
        d50_pf = m["p50"].predict(X_perf)
        d10_pf = m["p10"].predict(X_perf)
        d90_pf = m["p90"].predict(X_perf)
        p50_pf = np.maximum(350.0, base_d + d50_pf)
        p10_pf = np.maximum(350.0, np.minimum(p50_pf, base_d + d10_pf))
        p90_pf = np.maximum(p50_pf, base_d + d90_pf)

        for scen, lo, md, hi in [
            ("deterministic", p10_det, p50_det, p90_det),
            ("ensemble", p10_ens, p50_ens, p90_ens),
            ("perfect_wx", p10_pf, p50_pf, p90_pf),
        ]:
            chunks.append(
                pd.DataFrame(
                    {
                        "issued_at": issue_idx,
                        "target_timestamp": target_idx,
                        "horizon_h": h,
                        "scenario": scen,
                        "start_depth_mm": np.round(base_d, 1),
                        "p10_mm": np.round(lo, 1),
                        "p50_mm": np.round(md, 1),
                        "p90_mm": np.round(hi, 1),
                        "actual_depth_mm": np.round(act_d, 1),
                        "member_rain_48h_spread_mm": np.round(spread_48h, 2),
                    }
                )
            )

    df_all = pd.concat(chunks, ignore_index=True)
    out = Path("data/nwp_ensemble_backtest.parquet")
    df_all.to_parquet(out, index=False)
    summ = summarize(df_all)
    summ.to_csv(Path("data/nwp_ensemble_backtest_summary.csv"), index=False)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(summ.to_string(index=False))
    print(f"Completed full 2026 hourly backtest in {time.time() - t0:.1f}s → {out}")


if __name__ == "__main__":
    main()
