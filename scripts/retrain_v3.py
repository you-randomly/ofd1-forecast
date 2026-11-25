"""Offline Stage-B retraining script for OFD1 multi-horizon v3 models.

Fixes 2022 depth-logger -3h clock shift on the training copy, enforces 100%
train/serve parity via ``roll_future_features_for_horizon`` (using
``era5_soil_deep`` + ``ref_tank_168h`` moisture hygiene and early-window rainfall
sums), fits split-conformal dry-regime CQR band padding (``pad_dry_mm``)
strictly on pre-2026 data, and saves ``data/ofd1_models_v3.joblib``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.hydrology import (
    CAUSAL_COLS,
    HORIZONS,
    TRAIN_SPLIT_UTC,
    build_archived_nwp_trajectories,
    engineer_features,
    nwp_cols_for_horizon,
    roll_future_features_for_horizon,
)

SHORT_NWP_HORIZONS: set[int] = {2, 3, 6}
NWP_ARCHIVE_START_UTC = pd.Timestamp("2024-01-01 00:00:00+00:00")
MODERN_ERA_UTC = pd.Timestamp("2024-05-17 00:00:00+00:00")
CAL_SPLIT_UTC = pd.Timestamp("2025-07-01 00:00:00+00:00")


def _fit_horizon_quantiles(
    X: pd.DataFrame,
    y: np.ndarray,
    h: int,
    f_cols: list[str],
    nwp_cols: list[str],
    sample_w: np.ndarray,
) -> dict[str, HistGradientBoostingRegressor]:
    if h in SHORT_NWP_HORIZONS:
        nwp_set = set(nwp_cols)
        mono_cst: list[int] | None = [
            1 if (c in nwp_set and "tank" not in c) else 0 for c in f_cols
        ]
        p50_kw: dict[str, Any] = {
            "max_iter": 300,
            "learning_rate": 0.04,
            "max_depth": None,
            "max_leaf_nodes": 41,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
            "monotonic_cst": mono_cst,
            "random_state": 42,
        }
        q_kw: dict[str, Any] = {
            "max_iter": 220,
            "learning_rate": 0.04,
            "max_depth": None,
            "max_leaf_nodes": 41,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
            "monotonic_cst": mono_cst,
            "random_state": 42,
        }
    else:
        p50_kw = {
            "max_iter": 300,
            "learning_rate": 0.04,
            "max_depth": 6,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 25,
            "l2_regularization": 2.0,
            "monotonic_cst": None,
            "random_state": 42,
        }
        q_kw = {
            "max_iter": 220,
            "learning_rate": 0.04,
            "max_depth": 5,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
            "monotonic_cst": None,
            "random_state": 42,
        }

    return {
        "p50": HistGradientBoostingRegressor(loss="squared_error", **p50_kw).fit(
            X, y, sample_weight=sample_w
        ),
        "p10": HistGradientBoostingRegressor(loss="quantile", quantile=0.10, **q_kw).fit(
            X, y, sample_weight=sample_w
        ),
        "p90": HistGradientBoostingRegressor(loss="quantile", quantile=0.90, **q_kw).fit(
            X, y, sample_weight=sample_w
        ),
    }


def _compute_dry_cqr_padding(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    sub_idx: pd.DatetimeIndex,
    fut_mat: np.ndarray,
    ddepth_1h: np.ndarray,
    h: int,
    f_cols: list[str],
    nwp_cols: list[str],
    sample_w: np.ndarray,
) -> float:
    is_cal = sub_idx >= CAL_SPLIT_UTC
    is_dry_cal = (
        is_cal
        & (fut_mat[:, :h].sum(axis=1) < 2.0)
        & (np.abs(ddepth_1h) < 5.0)
    )
    if int(is_dry_cal.sum()) < 50 or int((~is_cal).sum()) < 500:
        return 0.0

    cal_models = _fit_horizon_quantiles(
        X_all[~is_cal],
        y_all[~is_cal],
        h=h,
        f_cols=f_cols,
        nwp_cols=nwp_cols,
        sample_w=sample_w[~is_cal],
    )
    q10_cal = cal_models["p10"].predict(X_all[is_dry_cal])
    q90_cal = cal_models["p90"].predict(X_all[is_dry_cal])
    y_dry = y_all[is_dry_cal]
    s_dry = np.maximum(q10_cal - y_dry, y_dry - q90_cal)
    return round(float(np.clip(np.quantile(s_dry, 0.80), 0.0, 22.0)), 2)


def train_v3_models(data_dir: Path, output_path: Path) -> dict[int, dict[str, Any]]:
    """Train and save the v3 multi-horizon quantile model bundle with dry CQR padding."""
    df_hourly = pd.read_parquet(data_dir / "ofd1_aligned_hourly.parquet").sort_index()
    if not isinstance(df_hourly.index, pd.DatetimeIndex):
        df_hourly.index = pd.to_datetime(df_hourly.index, utc=True)
    arch = pd.read_parquet(data_dir / "ofd1_nwp_archive.parquet").sort_index()
    if not isinstance(arch.index, pd.DatetimeIndex):
        arch.index = pd.to_datetime(arch.index, utc=True)

    # Fix 2022 depth logger -3h clock shift on training copy only
    df_train_in = df_hourly.copy()
    m22 = (
        (df_train_in.index >= pd.Timestamp("2022-01-01 00:00:00+00:00"))
        & (df_train_in.index < pd.Timestamp("2023-01-01 00:00:00+00:00"))
    )
    df_train_in.loc[m22, "depth_mm"] = df_train_in.loc[m22, "depth_mm"].shift(3)

    df_feat = engineer_features(df_train_in)
    train_obs = df_feat[df_feat.index < TRAIN_SPLIT_UTC].dropna(
        subset=["depth_mm", "ddepth_1h", "ddepth_24h"] + CAUSAL_COLS
    )
    tr_idx = train_obs.index

    rain_ref_s = df_feat["rain_mm"].fillna(df_feat["fused_rain_mm"]).fillna(0.0)
    era5_fut_mat = np.column_stack(
        [
            rain_ref_s.reindex(tr_idx + pd.Timedelta(hours=step), fill_value=0.0).values
            for step in range(1, 49)
        ]
    )
    nwp_mask = tr_idx >= NWP_ARCHIVE_START_UTC
    _, _, ens_p50_mat, _, _, _ = build_archived_nwp_trajectories(
        arch, df_feat, tr_idx[nwp_mask]
    )
    short_fut_mat = era5_fut_mat.copy()
    short_fut_mat[nwp_mask] = ens_p50_mat

    models_v3: dict[int, dict[str, Any]] = {}
    for h in HORIZONS:
        target_ts = tr_idx + pd.Timedelta(hours=h)
        tgt_depth = df_feat["depth_mm"].reindex(target_ts).values
        ok = (target_ts < TRAIN_SPLIT_UTC) & (~np.isnan(tgt_depth))
        sub_idx = tr_idx[ok]
        y_all = tgt_depth[ok] - df_feat.loc[sub_idx, "depth_mm"].values
        fut_mat = short_fut_mat[ok] if h in SHORT_NWP_HORIZONS else era5_fut_mat[ok]

        nwp_cols = nwp_cols_for_horizon(h)
        f_cols = CAUSAL_COLS + nwp_cols
        X_all = roll_future_features_for_horizon(df_feat, sub_idx, fut_mat, h)[f_cols]

        modern_weight = 2.5 if h == 1 else 1.5
        sample_w = np.where(sub_idx >= MODERN_ERA_UTC, modern_weight, 1.0)
        dd1_sub = df_feat.loc[sub_idx, "ddepth_1h"].values

        pad_dry_mm = _compute_dry_cqr_padding(
            X_all=X_all,
            y_all=y_all,
            sub_idx=sub_idx,
            fut_mat=fut_mat,
            ddepth_1h=dd1_sub,
            h=h,
            f_cols=f_cols,
            nwp_cols=nwp_cols,
            sample_w=sample_w,
        )
        fitted = _fit_horizon_quantiles(
            X_all, y_all, h=h, f_cols=f_cols, nwp_cols=nwp_cols, sample_w=sample_w
        )
        models_v3[h] = {
            "p10": fitted["p10"],
            "p50": fitted["p50"],
            "p90": fitted["p90"],
            "pad_dry_mm": pad_dry_mm,
        }
        regime = f"archived-NWP(w={modern_weight})" if h in SHORT_NWP_HORIZONS else f"hindsight(w={modern_weight})"
        print(f"Trained v3 +{h:2d}h [{regime}] on {len(sub_idx)} rows | pad_dry_mm={pad_dry_mm:.2f}")

    joblib.dump(models_v3, output_path, compress=3)
    print(f"Saved v3 model bundle to {output_path}")
    return models_v3


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain OFD1 v3 multi-horizon models.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output or (args.data_dir / "ofd1_models_v3.joblib")
    train_v3_models(args.data_dir, out)


if __name__ == "__main__":
    main()
