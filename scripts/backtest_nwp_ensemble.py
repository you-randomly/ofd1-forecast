"""Fast, exact out-of-sample 2026 backtest of the 6-model NWP ensemble vs. deterministic NWP.

Evaluates every issue hour in the 2026 unseen test set (models trained strictly on targets < 2026-01-01)
across three issue-time weather scenarios:

* ``deterministic`` – Open-Meteo ``best_match`` rain as forecast at issue time, hydrological P10–P90 band only.
* ``ensemble``      – 6-model NWP ensemble (UKMO, ECMWF, ICON, AROME, GFS, Blend) as forecast at issue time,
  volume-conserved P10/P50/P90 rain trajectories + quadrature-combined P10–P90 band.
* ``perfect_wx``    – Observed/reanalysis rain (reference upper bound).

Writes ``data/nwp_ensemble_backtest.parquet`` and ``data/nwp_ensemble_backtest_summary.csv`` when not in
``--read-only`` mode, and verifies or updates ``benchmarks/skill_ledger.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.engine import OFD1ForecastEngine
from app.hydrology import (
    CAUSAL_COLS,
    ELEVATED_DEPTH_MM,
    EVAL_SKILL_HORIZONS,
    NWP_MODELS,
    RED_DEPTH_MM,
    TRAIN_SPLIT_UTC,
    apply_hydrological_guardrails,
    build_archived_nwp_trajectories,
    combine_ensemble_quantile_deltas,
    reconstruct_depth_quantiles,
    roll_future_features_for_horizon,
)

MODELS = NWP_MODELS
EVAL_HORIZONS = EVAL_SKILL_HORIZONS

FINGERPRINT_FILES: tuple[str, ...] = (
    "app/hydrology.py",
    "app/engine.py",
    "scripts/backtest_nwp_ensemble.py",
    "data/ofd1_models_v2.joblib",
    "data/ofd1_aligned_hourly.parquet",
    "data/ofd1_nwp_archive.parquet",
)

DEFAULT_TOLERANCES: dict[str, float] = {
    "max_mae_regression_mm": 0.5,
    "max_rmse_regression_mm": 0.75,
    "min_coverage_80_pct": 66.0,
    "max_f1_regression": 0.015,
}


def compute_skill_fingerprint(root_dir: Path | str = REPO_ROOT) -> dict[str, Any]:
    """Compute standalone per-file and composite SHA-256 digests over FINGERPRINT_FILES."""
    root = Path(root_dir)
    file_hashes: dict[str, str] = {}
    composite = hashlib.sha256()
    for rel_path in FINGERPRINT_FILES:
        full_path = root / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Missing required fingerprint file: {full_path}")
        digest = hashlib.sha256(full_path.read_bytes()).hexdigest()
        file_hashes[rel_path] = digest
        composite.update(f"{rel_path}:{digest}\n".encode())
    return {
        "fingerprint_sha256": composite.hexdigest(),
        "files": file_hashes,
    }


def _build_future_rain_trajectories(
    arch: pd.DataFrame, valid_obs: pd.DataFrame, issue_idx: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Delegate archived NWP trajectory construction to app.hydrology.build_archived_nwp_trajectories."""
    return build_archived_nwp_trajectories(arch, valid_obs, issue_idx)


def _roll_future_features_for_horizon(
    valid_obs: pd.DataFrame,
    issue_idx: pd.DatetimeIndex,
    fut_rain_mat: np.ndarray,
    h: int,
) -> pd.DataFrame:
    """Delegate future feature rolling to app.hydrology.roll_future_features_for_horizon."""
    return roll_future_features_for_horizon(valid_obs, issue_idx, fut_rain_mat, h)


def _binary_f1(pred: pd.Series | np.ndarray, obs: pd.Series | np.ndarray) -> float:
    pred_b = np.asarray(pred, dtype=bool)
    obs_b = np.asarray(obs, dtype=bool)
    tp = int((pred_b & obs_b).sum())
    fp = int((pred_b & ~obs_b).sum())
    fn = int((~pred_b & obs_b).sum())
    denom = 2 * tp + fp + fn
    return float(2.0 * tp / denom) if denom > 0 else 0.0


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["actual_depth_mm"]).copy()
    y, lo, md, hi = df["actual_depth_mm"], df["p10_mm"], df["p50_mm"], df["p90_mm"]
    alpha = 0.2
    df["covered"] = (y >= lo) & (y <= hi)
    df["width"] = hi - lo
    df["err"] = md - y
    df["abs_err"] = df["err"].abs()
    df["sq_err"] = df["err"] ** 2
    df["interval_score"] = (
        (hi - lo) + (2 / alpha) * (lo - y).clip(lower=0) + (2 / alpha) * (y - hi).clip(lower=0)
    )
    df["obs_600"] = y >= ELEVATED_DEPTH_MM
    df["p90_600"] = hi >= ELEVATED_DEPTH_MM
    df["p50_600"] = md >= ELEVATED_DEPTH_MM
    df["obs_red"] = y >= RED_DEPTH_MM
    df["p90_red"] = hi >= RED_DEPTH_MM
    df["p50_red"] = md >= RED_DEPTH_MM

    def _agg(g: pd.DataFrame) -> pd.Series:
        obs_red = g["obs_red"]
        obs_600 = g["obs_600"]
        wet = g[g["member_rain_48h_spread_mm"] >= 2.0]
        out: dict[str, Any] = {
            "n": len(g),
            "coverage_pct": 100 * g["covered"].mean(),
            "wet_coverage_pct": 100 * wet["covered"].mean() if len(wet) else np.nan,
            "mean_width_mm": g["width"].mean(),
            "wet_width_mm": wet["width"].mean() if len(wet) else np.nan,
            "mae_p50_mm": g["abs_err"].mean(),
            "rmse_p50_mm": float(np.sqrt(g["sq_err"].mean())),
            "interval_score": g["interval_score"].mean(),
            "obs_red_hours": int(obs_red.sum()),
        }
        for q in ("p90", "p50"):
            pred = g[f"{q}_red"]
            hits = int((pred & obs_red).sum())
            fa = int((pred & ~obs_red).sum())
            miss = int((~pred & obs_red).sum())
            out[f"{q}_red_pod_pct"] = 100 * hits / max(1, hits + miss)
            out[f"{q}_red_far_pct"] = 100 * fa / max(1, hits + fa)
            out[f"{q}_red_false_alarms"] = fa
            out[f"{q}_red_misses"] = miss

        out["f1_600_p50"] = _binary_f1(g["p50_600"], obs_600)
        out["f1_600_p90"] = _binary_f1(g["p90_600"], obs_600)
        out["f1_700_p50"] = _binary_f1(g["p50_red"], obs_red)
        out["f1_700_p90"] = _binary_f1(g["p90_red"], obs_red)
        return pd.Series(out)

    summ = df.groupby(["horizon_h", "scenario"]).apply(_agg, include_groups=False).reset_index()
    f1_cols = ["f1_600_p50", "f1_600_p90", "f1_700_p50", "f1_700_p90"]
    other_float_cols = [
        c for c in summ.select_dtypes(include=[np.floating, float]).columns if c not in f1_cols
    ]
    summ[other_float_cols] = summ[other_float_cols].round(2)
    summ[f1_cols] = summ[f1_cols].round(4)
    for int_col in (
        "n",
        "obs_red_hours",
        "p90_red_false_alarms",
        "p90_red_misses",
        "p50_red_false_alarms",
        "p50_red_misses",
    ):
        if int_col in summ.columns:
            summ[int_col] = summ[int_col].astype(int)
    return summ


def run_backtest(
    data_dir: Path | str = "data",
    read_only: bool = False,
    horizons: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run vectorized 2026 out-of-sample backtest across horizons and scenarios."""
    d_dir = Path(data_dir)
    eval_h = horizons if horizons is not None else EVAL_HORIZONS

    eng = OFD1ForecastEngine(data_dir=d_dir, read_only=True)
    eng.df_hourly = pd.read_parquet(eng.aligned_1h_path).sort_index()
    eng.models = joblib.load(eng.models_path)
    df_feat = eng._engineer_features(eng.df_hourly)
    valid_obs = df_feat.dropna(subset=["depth_mm"] + CAUSAL_COLS).sort_index()
    arch = pd.read_parquet(d_dir / "ofd1_nwp_archive.parquet").sort_index()

    end_ts = min(valid_obs.index.max(), arch.index.max()) - pd.Timedelta(hours=48)
    issue_idx = valid_obs.loc[TRAIN_SPLIT_UTC:end_ts].index

    det_r, p10_r, p50_r, p90_r, perf_r, spread_48h = _build_future_rain_trajectories(
        arch, valid_obs, issue_idx
    )
    base_d = valid_obs.loc[issue_idx, "depth_mm"].values
    fused_3h = (
        valid_obs.loc[issue_idx, "fused_rain_sum_3h"].values
        if "fused_rain_sum_3h" in valid_obs.columns
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

    chunks = []
    for h in eval_h:
        X_det = _roll_future_features_for_horizon(valid_obs, issue_idx, det_r, h)
        X_p10 = _roll_future_features_for_horizon(valid_obs, issue_idx, p10_r, h)
        X_p50 = _roll_future_features_for_horizon(valid_obs, issue_idx, p50_r, h)
        X_p90 = _roll_future_features_for_horizon(valid_obs, issue_idx, p90_r, h)
        X_perf = _roll_future_features_for_horizon(valid_obs, issue_idx, perf_r, h)

        m = eng.models[h]
        target_idx = issue_idx + pd.Timedelta(hours=h)
        act_d = valid_obs["depth_mm"].reindex(target_idx).values

        # 1. Deterministic NWP (hydrological P10-P90 with physical guardrails)
        d50_det = m["p50"].predict(X_det)
        d10_det = m["p10"].predict(X_det)
        d90_det = m["p90"].predict(X_det)
        d10_det_c, d50_det_c, d90_det_c = apply_hydrological_guardrails(
            base_depth=base_d,
            ddepth_1h=dd_1h,
            fused_rain_3h=fused_3h,
            moisture_index=moist_idx,
            tank_168h=t_168,
            horizon_h=h,
            fut_rain_sum=X_det["fut_rain_sum_6h_at_target"].values,
            d_p10=d10_det,
            d_p50=d50_det,
            d_p90=d90_det,
        )
        p10_det, p50_det, p90_det = reconstruct_depth_quantiles(
            base_depth=base_d,
            d_p10_c=d10_det_c,
            d_p50_c=d50_det_c,
            d_p90_c=d90_det_c,
        )

        # 2. 6-Model NWP Ensemble (quadrature combination + physical guardrails matching _forecast_anchors)
        d_p50 = m["p50"].predict(X_p50)
        d_p10_on_p50 = m["p10"].predict(X_p50)
        d_p50_on_p10 = m["p50"].predict(X_p10)
        d_p90_on_p50 = m["p90"].predict(X_p50)
        d_p90_on_p90 = m["p90"].predict(X_p90)
        d_p50_on_p90 = m["p50"].predict(X_p90)

        d10_rel, d50_rel, d90_rel, _ = combine_ensemble_quantile_deltas(
            d_p50=d_p50,
            d_p10_on_p50=d_p10_on_p50,
            d_p50_on_p10=d_p50_on_p10,
            d_p90_on_p50=d_p90_on_p50,
            d_p90_on_p90=d_p90_on_p90,
            d_p50_on_p90=d_p50_on_p90,
        )
        d10_ens_c, d50_ens_c, d90_ens_c = apply_hydrological_guardrails(
            base_depth=base_d,
            ddepth_1h=dd_1h,
            fused_rain_3h=fused_3h,
            moisture_index=moist_idx,
            tank_168h=t_168,
            horizon_h=h,
            fut_rain_sum=X_p50["fut_rain_sum_6h_at_target"].values,
            d_p10=d10_rel,
            d_p50=d50_rel,
            d_p90=d90_rel,
        )
        p10_ens, p50_ens, p90_ens = reconstruct_depth_quantiles(
            base_depth=base_d,
            d_p10_c=d10_ens_c,
            d_p50_c=d50_ens_c,
            d_p90_c=d90_ens_c,
        )

        # 3. Perfect Wx reference (with physical guardrails)
        d50_pf = m["p50"].predict(X_perf)
        d10_pf = m["p10"].predict(X_perf)
        d90_pf = m["p90"].predict(X_perf)
        d10_pf_c, d50_pf_c, d90_pf_c = apply_hydrological_guardrails(
            base_depth=base_d,
            ddepth_1h=dd_1h,
            fused_rain_3h=fused_3h,
            moisture_index=moist_idx,
            tank_168h=t_168,
            horizon_h=h,
            fut_rain_sum=X_perf["fut_rain_sum_6h_at_target"].values,
            d_p10=d10_pf,
            d_p50=d50_pf,
            d_p90=d90_pf,
        )
        p10_pf, p50_pf, p90_pf = reconstruct_depth_quantiles(
            base_depth=base_d,
            d_p10_c=d10_pf_c,
            d_p50_c=d50_pf_c,
            d_p90_c=d90_pf_c,
        )

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
    summ = summarize(df_all)

    if not read_only:
        df_all.to_parquet(d_dir / "nwp_ensemble_backtest.parquet", index=False)
        summ.to_csv(d_dir / "nwp_ensemble_backtest_summary.csv", index=False)

    return df_all, summ


def _extract_ledger_metrics(summary_df: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    metrics: dict[str, dict[str, dict[str, float]]] = {"ensemble": {}, "deterministic": {}}
    metric_cols = [
        "mae_p50_mm",
        "rmse_p50_mm",
        "coverage_pct",
        "f1_600_p50",
        "f1_600_p90",
        "f1_700_p50",
        "f1_700_p90",
    ]
    for scen in ("ensemble", "deterministic"):
        sub = summary_df[summary_df["scenario"] == scen]
        for _, row in sub.iterrows():
            h_key = str(int(row["horizon_h"]))
            metrics[scen][h_key] = {col: float(row[col]) for col in metric_cols}
    return metrics


def verify_or_update_skill_ledger(
    summary_df: pd.DataFrame,
    ledger_path: Path = REPO_ROOT / "benchmarks" / "skill_ledger.json",
    root_dir: Path = REPO_ROOT,
    update: bool = False,
    reset_baseline: bool = False,
) -> dict[str, Any]:
    """Validate summary_df against benchmarks/skill_ledger.json or update the ledger."""
    fp = compute_skill_fingerprint(root_dir)
    current_metrics = _extract_ledger_metrics(summary_df)

    existing: dict[str, Any] | None = None
    if ledger_path.exists():
        existing = json.loads(ledger_path.read_text(encoding="utf-8"))

    if existing is not None and not reset_baseline:
        if not update and existing.get("current_fingerprint_sha256") != fp["fingerprint_sha256"]:
            raise AssertionError(
                f"Skill ledger fingerprint mismatch: ledger has {existing.get('current_fingerprint_sha256')}, "
                f"live workspace has {fp['fingerprint_sha256']}. "
                "Run `uv run python scripts/backtest_nwp_ensemble.py --update-ledger`."
            )
        tolerances = existing.get("tolerances", DEFAULT_TOLERANCES)
        baseline = existing.get("baseline_metrics", {})
        failures: list[str] = []
        for scen in ("ensemble", "deterministic"):
            base_scen = baseline.get(scen, {})
            curr_scen = current_metrics.get(scen, {})
            for h_key, b_vals in base_scen.items():
                if h_key not in curr_scen:
                    failures.append(f"Missing horizon +{h_key}h for scenario '{scen}'")
                    continue
                c_vals = curr_scen[h_key]
                if c_vals["mae_p50_mm"] > b_vals["mae_p50_mm"] + tolerances["max_mae_regression_mm"]:
                    failures.append(
                        f"{scen} +{h_key}h MAE regressed: {c_vals['mae_p50_mm']:.2f} > "
                        f"{b_vals['mae_p50_mm']:.2f} + {tolerances['max_mae_regression_mm']}"
                    )
                if c_vals["rmse_p50_mm"] > b_vals["rmse_p50_mm"] + tolerances["max_rmse_regression_mm"]:
                    failures.append(
                        f"{scen} +{h_key}h RMSE regressed: {c_vals['rmse_p50_mm']:.2f} > "
                        f"{b_vals['rmse_p50_mm']:.2f} + {tolerances['max_rmse_regression_mm']}"
                    )
                min_cov = max(float(tolerances["min_coverage_80_pct"]), b_vals["coverage_pct"] - 1.5)
                if scen == "ensemble" and c_vals["coverage_pct"] < min_cov:
                    failures.append(
                        f"{scen} +{h_key}h coverage regressed: {c_vals['coverage_pct']:.2f}% < {min_cov:.2f}%"
                    )
                if c_vals["f1_600_p50"] < b_vals["f1_600_p50"] - tolerances["max_f1_regression"]:
                    failures.append(
                        f"{scen} +{h_key}h F1@600(P50) regressed: {c_vals['f1_600_p50']:.4f} < "
                        f"{b_vals['f1_600_p50']:.4f} - {tolerances['max_f1_regression']}"
                    )
                if c_vals["f1_700_p90"] < b_vals["f1_700_p90"] - tolerances["max_f1_regression"]:
                    failures.append(
                        f"{scen} +{h_key}h F1@700(P90) regressed: {c_vals['f1_700_p90']:.4f} < "
                        f"{b_vals['f1_700_p90']:.4f} - {tolerances['max_f1_regression']}"
                    )
        if failures:
            raise AssertionError("Forecast skill regression gate failed:\n  - " + "\n  - ".join(failures))
    elif not update:
        raise FileNotFoundError(
            f"Skill ledger not found at {ledger_path}. Run with --update-ledger to initialize."
        )

    if update:
        now_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        baseline_to_save = (
            current_metrics
            if (existing is None or reset_baseline)
            else existing.get("baseline_metrics", current_metrics)
        )
        tolerances_to_save = (
            DEFAULT_TOLERANCES
            if (existing is None or reset_baseline)
            else existing.get("tolerances", DEFAULT_TOLERANCES)
        )
        history = list(existing.get("history", [])) if existing is not None else []
        # Avoid duplicate consecutive history entries for the identical fingerprint
        if not history or history[-1].get("fingerprint_sha256") != fp["fingerprint_sha256"]:
            history.append(
                {
                    "timestamp_utc": now_iso,
                    "fingerprint_sha256": fp["fingerprint_sha256"],
                    "ensemble_metrics": current_metrics["ensemble"],
                }
            )
        else:
            history[-1] = {
                "timestamp_utc": now_iso,
                "fingerprint_sha256": fp["fingerprint_sha256"],
                "ensemble_metrics": current_metrics["ensemble"],
            }

        payload = {
            "schema_version": 1,
            "generated_at_utc": now_iso,
            "current_fingerprint_sha256": fp["fingerprint_sha256"],
            "files": fp["files"],
            "horizons_evaluated": EVAL_HORIZONS,
            "tolerances": tolerances_to_save,
            "baseline_metrics": baseline_to_save,
            "current_metrics": current_metrics,
            "history": history[-20:],
        }
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload

    return existing or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="OFD1 2026 Out-of-Sample NWP Ensemble Backtest & Skill Gate")
    parser.add_argument("--data-dir", default="data", help="Path to telemetry/model data directory")
    parser.add_argument("--read-only", action="store_true", help="Do not write Parquet/CSV files to data/")
    parser.add_argument(
        "--verify-ledger",
        action="store_true",
        help="Verify SHA-256 fingerprint and non-regression against benchmarks/skill_ledger.json",
    )
    parser.add_argument(
        "--update-ledger",
        action="store_true",
        help="Update benchmarks/skill_ledger.json with current fingerprint and metrics",
    )
    parser.add_argument(
        "--reset-baseline",
        action="store_true",
        help="Reset baseline_metrics in benchmarks/skill_ledger.json when used with --update-ledger",
    )
    args = parser.parse_args()

    t0 = time.time()
    data_dir = Path(args.data_dir)
    summary_csv_path = data_dir / "nwp_ensemble_backtest_summary.csv"

    # Fast path for --verify-ledger --read-only when summary CSV already exists and fingerprint matches
    if args.verify_ledger and args.read_only and not args.update_ledger and summary_csv_path.exists():
        summ = pd.read_csv(summary_csv_path)
        expected_cols = {"rmse_p50_mm", "f1_600_p50", "f1_600_p90", "f1_700_p50", "f1_700_p90"}
        if expected_cols.issubset(set(summ.columns)) and set(EVAL_HORIZONS).issubset(
            set(summ["horizon_h"].astype(int).unique())
        ):
            verify_or_update_skill_ledger(summ, update=False, reset_baseline=False)
            print(
                f"Verified skill ledger and 6-file SHA-256 fingerprint against {summary_csv_path} in {time.time() - t0:.2f}s"
            )
            return

    _, summ = run_backtest(data_dir=data_dir, read_only=args.read_only)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(summ.to_string(index=False))
    print(f"Completed full 2026 hourly backtest in {time.time() - t0:.1f}s")

    if args.verify_ledger or args.update_ledger:
        verify_or_update_skill_ledger(
            summ,
            update=args.update_ledger,
            reset_baseline=args.reset_baseline,
        )
        if args.update_ledger:
            print("Updated benchmarks/skill_ledger.json")
        else:
            print("Verified benchmarks/skill_ledger.json")


if __name__ == "__main__":
    main()
