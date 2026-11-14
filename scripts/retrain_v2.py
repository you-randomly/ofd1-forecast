"""Offline Stage-2 retraining script for OFD1 multi-horizon v2 models.

Short horizons (1h, 2h, 3h, 6h) replace 2024-2025 hindsight future rainfall
with archived 6-model NWP median (ens_p50) trajectories from
data/ofd1_nwp_archive.parquet (modern-era sample weight 1.5), while longer
horizons (12h..48h) preserve the v1 hindsight training regime to avoid
NWP-smoothing degradation at medium range.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    attach_horizon_future_cols,
    build_archived_nwp_trajectories,
    engineer_features,
    nwp_cols_for_horizon,
    roll_future_features_for_horizon,
)

SHORT_NWP_HORIZONS: set[int] = {1, 2, 3, 6}
NWP_ARCHIVE_START_UTC = pd.Timestamp("2024-01-01 00:00:00+00:00")
MODERN_ERA_UTC = pd.Timestamp("2024-05-17 00:00:00+00:00")


def _fit_horizon_quantiles(
    X: pd.DataFrame,
    y: np.ndarray,
    f_cols: list[str],
    nwp_cols: list[str],
    sample_w: np.ndarray,
) -> dict[str, HistGradientBoostingRegressor]:
    nwp_set = set(nwp_cols)
    mono_cst = [1 if c in nwp_set else 0 for c in f_cols]
    common_kw = {
        "max_iter": 300,
        "learning_rate": 0.04,
        "max_leaf_nodes": 41,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "monotonic_cst": mono_cst,
        "random_state": 42,
    }
    return {
        "p50": HistGradientBoostingRegressor(loss="squared_error", **common_kw).fit(
            X, y, sample_weight=sample_w
        ),
        "p10": HistGradientBoostingRegressor(loss="quantile", quantile=0.10, **common_kw).fit(
            X, y, sample_weight=sample_w
        ),
        "p90": HistGradientBoostingRegressor(loss="quantile", quantile=0.90, **common_kw).fit(
            X, y, sample_weight=sample_w
        ),
    }


def train_v2_models(data_dir: Path, output_path: Path) -> dict[int, dict[str, HistGradientBoostingRegressor]]:
    """Train and save the hybrid v2 multi-horizon model bundle.

    Starts from data/ofd1_models_v1.joblib when present so +1h, +12h..+48h, and
    calibrated P10/P90 quantile bounds remain regression-free, while upgrading
    short-range P50 (+2h, +3h, +6h) and +6h quantiles on 2024-2025 archived NWP.
    """
    df_hourly = pd.read_parquet(data_dir / "ofd1_aligned_hourly.parquet").sort_index()
    if not isinstance(df_hourly.index, pd.DatetimeIndex):
        df_hourly.index = pd.to_datetime(df_hourly.index, utc=True)
    arch = pd.read_parquet(data_dir / "ofd1_nwp_archive.parquet").sort_index()
    if not isinstance(arch.index, pd.DatetimeIndex):
        arch.index = pd.to_datetime(arch.index, utc=True)

    df_feat = engineer_features(df_hourly)
    train_obs = df_feat[df_feat.index < TRAIN_SPLIT_UTC].dropna(
        subset=["depth_mm", "ddepth_1h", "ddepth_24h"]
    )
    nwp_issue_idx = train_obs[train_obs.index >= NWP_ARCHIVE_START_UTC].index
    _, _, ens_p50_mat, _, _, _ = build_archived_nwp_trajectories(arch, df_feat, nwp_issue_idx)

    v1_path = data_dir / "ofd1_models_v1.joblib"
    models_v2: dict[int, dict[str, HistGradientBoostingRegressor]] = (
        joblib.load(v1_path) if v1_path.exists() else {}
    )
    target_horizons = [2, 3, 6] if models_v2 else list(HORIZONS)
    for h in target_horizons:
        work = attach_horizon_future_cols(df_feat, h)
        work["target_delta"] = work["depth_mm"].shift(-h) - work["depth_mm"]
        nwp_cols = nwp_cols_for_horizon(h)
        f_cols = CAUSAL_COLS + nwp_cols

        if h in SHORT_NWP_HORIZONS:
            nwp_fut = roll_future_features_for_horizon(df_feat, nwp_issue_idx, ens_p50_mat, h)[nwp_cols]
            for col in nwp_cols:
                work.loc[nwp_issue_idx, col] = nwp_fut[col].to_numpy(dtype=float)
            modern_weight = 1.5
        else:
            modern_weight = 2.5

        target_ts = work.index + pd.Timedelta(hours=h)
        valid = work[target_ts < TRAIN_SPLIT_UTC].dropna(
            subset=["target_delta", "depth_mm", "ddepth_1h", "ddepth_24h"] + f_cols
        )
        sample_w = np.where(valid.index >= MODERN_ERA_UTC, modern_weight, 1.0)
        fitted = _fit_horizon_quantiles(
            valid[f_cols], valid["target_delta"].values, f_cols, nwp_cols, sample_w
        )
        if h in {2, 3} and h in models_v2:
            models_v2[h]["p50"] = fitted["p50"]
        else:
            models_v2[h] = fitted
        regime = f"archived-NWP(w={modern_weight})" if h in SHORT_NWP_HORIZONS else f"hindsight(w={modern_weight})"
        print(f"Trained v2 +{h}h [{regime}] on {len(valid)} rows")

    joblib.dump(models_v2, output_path, compress=3)
    print(f"Saved v2 model bundle to {output_path}")
    return models_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain OFD1 v2 multi-horizon models.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output or (args.data_dir / "ofd1_models_v2.joblib")
    train_v2_models(args.data_dir, out)


if __name__ == "__main__":
    main()
