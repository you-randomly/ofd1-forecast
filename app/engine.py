from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

logger = logging.getLogger("ofd1.engine")

HORIZONS = [1, 2, 3, 6, 12, 18, 24, 36, 48]
STEP_DATUM_MM = 550.0
AMBER_DEPTH_MM = 620.0
RED_DEPTH_MM = 700.0  # ~6 inches over The Step
TRAIN_SPLIT_UTC = pd.Timestamp("2026-01-01 00:00:00+00:00")

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
    + [f"fused_rain_lag_{k}h" for k in range(0, 7)]
    + [f"ref_rain_lag_{k}h" for k in range(0, 7)]
    + [f"eff_rain_lag_{k}h" for k in range(0, 7)]
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
    for k in range(0, min(h, 6)):
        cols.append(f"fut_rain_minus_{k}h_from_target")
    return cols


class OFD1ForecastEngine:
    """Manages telemetry synchronization, feature engineering, and multi-horizon P10/P50/P90 inference."""

    def __init__(self, data_dir: Path | str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.aligned_1h_path = self.data_dir / "ofd1_aligned_hourly.parquet"
        self.raw_ts_path = self.data_dir / "ofd1_thingspeak_raw.parquet"
        self.models_path = self.data_dir / "ofd1_models_v1.joblib"
        self.metrics_path = self.data_dir / "phase2_trial_metrics.parquet"
        self.breakdown_path = self.data_dir / "horizon_breakdown_1h_to_48h.parquet"
        self.preds_backtest_path = self.data_dir / "phase2_trial_predictions.parquet"
        self.ledger_path = self.data_dir / "forecast_verification_ledger.parquet"

        self.models: dict[int, dict[str, HistGradientBoostingRegressor]] = {}
        self.df_hourly: pd.DataFrame = pd.DataFrame()
        self.df_future_wx: pd.DataFrame = pd.DataFrame()
        self.last_sync_utc: str | None = None
        self.sync_status: str = "initializing"
        self.latest_payload: dict[str, Any] = {}

    def initialize(self) -> None:
        """Load cached hourly dataset and load or train the multi-horizon quantile models."""
        seed_dir = Path(os.environ.get("OFD1_SEED_DIR", "seed_data"))
        if seed_dir.exists() and seed_dir.resolve() != self.data_dir.resolve():
            for item in seed_dir.glob("*"):
                dest = self.data_dir / item.name
                if not dest.exists():
                    shutil.copy2(item, dest)
                    logger.info("Seeded PVC data file %s from %s", dest, item)

        if not self.aligned_1h_path.exists():
            raise FileNotFoundError(f"Missing aligned telemetry dataset at {self.aligned_1h_path}")

        self.df_hourly = pd.read_parquet(self.aligned_1h_path).sort_index()
        hist_csv_path = self.data_dir / "historic_ofd_streamway_cleaned.csv"
        hist_wx_path = self.data_dir / "open_meteo_2015_2023.parquet"
        if (
            hist_csv_path.exists()
            and hist_wx_path.exists()
            and self.df_hourly.index.min().year > 2016
        ):
            logger.info("Merging 2015–2023 historical streamway archive into %s...", self.aligned_1h_path)
            df_h = pd.read_csv(hist_csv_path)
            df_h["timestamp"] = pd.to_datetime(
                df_h["Date and Time"], format="%m/%d/%y %H:%M", errors="coerce", utc=True
            )
            df_h = (
                df_h.dropna(subset=["timestamp"])
                .sort_values("timestamp")
                .drop_duplicates("timestamp")
            )
            df_h["depth_raw"] = pd.to_numeric(df_h["Depth (mm)"], errors="coerce")
            df_h["swcc_rain_mm"] = pd.to_numeric(df_h["Rainfall (mm)"], errors="coerce")
            bad_depth = (df_h["depth_raw"] < 300.0) | (df_h["depth_raw"] > 3500.0)
            df_h.loc[bad_depth, "depth_raw"] = np.nan
            df_h = df_h.set_index("timestamp")
            first_ts = df_h["depth_raw"].first_valid_index().floor("1h")
            last_ts = df_h["depth_raw"].last_valid_index().floor("1h")
            df_h = df_h.loc[first_ts:last_ts]

            df_hist_1h = pd.DataFrame(
                index=pd.date_range(first_ts, last_ts, freq="1h", tz="UTC")
            )
            df_hist_1h.index.name = "timestamp"
            df_hist_1h["depth_mm_raw"] = df_h["depth_raw"].resample("1h").mean()
            df_hist_1h["depth_mm_unaligned"] = df_hist_1h["depth_mm_raw"].interpolate(
                method="time", limit=3
            )
            df_hist_1h["swcc_rain_mm"] = df_h["swcc_rain_mm"].resample("1h").sum(min_count=1)
            df_hist_1h["stump_rain_mm"] = np.nan
            df_hist_1h["swcc_press_hpa"] = np.nan
            df_hist_1h["stump_press_hpa"] = np.nan

            rolling_base = (
                df_hist_1h["depth_mm_unaligned"]
                .rolling(1440, min_periods=120, center=True)
                .quantile(0.05)
                .bfill()
                .ffill()
            )
            df_hist_1h["depth_mm"] = (df_hist_1h["depth_mm_unaligned"] - rolling_base) + 535.0
            df_wx_hist = pd.read_parquet(hist_wx_path)
            df_hist_merged = df_hist_1h.drop(columns=["depth_mm_unaligned"]).join(
                df_wx_hist, how="inner"
            )
            common_cols = [
                "depth_mm_raw",
                "depth_mm",
                "swcc_rain_mm",
                "stump_rain_mm",
                "swcc_press_hpa",
                "stump_press_hpa",
                "rain_mm",
                "temp_c",
                "wind_kmh",
                "era5_press_hpa",
                "era5_soil_moisture",
                "era5_soil_deep",
                "et0_mm",
            ]
            self.df_hourly = pd.concat(
                [df_hist_merged[common_cols], self.df_hourly[common_cols]], axis=0
            )
            self.df_hourly = self.df_hourly[
                ~self.df_hourly.index.duplicated(keep="last")
            ].sort_index()
            self.df_hourly.to_parquet(self.aligned_1h_path)

        df_feat = self._engineer_features(self.df_hourly)

        if self.models_path.exists():
            try:
                self.models = joblib.load(self.models_path)
                logger.info("Loaded cached multi-horizon models from %s", self.models_path)
            except Exception as exc:
                logger.warning("Failed to load cached models (%s), retraining...", exc)
                self._train_and_save_models(df_feat)
        else:
            self._train_and_save_models(df_feat)

        # Build initial payload from cached data before network sync
        self.latest_payload = self._build_dashboard_payload(df_feat, self.df_future_wx)
        self._ensure_and_update_verification_ledger(df_feat)
        self.sync_status = "ready"

    def _engineer_features(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy().sort_index()

        # 1. Glitch-filtered Stump rain
        stump_glitch = (df["stump_rain_mm"] >= 15.0) & (
            (df["swcc_rain_mm"].fillna(0) < 5.0) & (df["rain_mm"].fillna(0) < 5.0)
        )
        df["stump_glitch_flag"] = stump_glitch
        df["stump_rain_clean"] = df["stump_rain_mm"].mask(stump_glitch, np.nan)
        df["club_rain_mean"] = df[["stump_rain_clean", "swcc_rain_mm"]].mean(axis=1)
        df["fused_rain_mm"] = df["club_rain_mean"].fillna(df["rain_mm"]).fillna(0.0)

        wind_factor = 1.0 + 0.004 * (df["wind_kmh"].fillna(15.0) - 15.0).clip(lower=0.0)
        df["fused_rain_wind_adj"] = df["fused_rain_mm"] * wind_factor

        # 2. Rolling 30-day (720h) 5th-percentile baseflow datum
        df["baseflow_30d"] = df["depth_mm"].rolling(720, min_periods=24).quantile(0.05).bfill()
        df["stage_above_base"] = df["depth_mm"] - df["baseflow_30d"]

        # 3. Antecedent moisture tanks
        et0 = df["et0_mm"].fillna(0.05)
        net_rain = (df["fused_rain_mm"] - 0.8 * et0).clip(lower=0.0)
        rain_ref = df["rain_mm"].fillna(df["fused_rain_mm"]).fillna(0.0)

        for hl in [3, 6, 12, 24, 96, 168]:
            alpha = 1.0 - np.exp(-np.log(2) / hl)
            df[f"tank_{hl}h"] = net_rain.ewm(alpha=alpha, adjust=False).mean()
            df[f"ref_tank_{hl}h"] = rain_ref.ewm(alpha=alpha, adjust=False).mean()

        soil_min = df["era5_soil_moisture"].expanding(min_periods=48).min().fillna(0.30)
        soil_max = df["era5_soil_moisture"].expanding(min_periods=48).max().fillna(0.43)
        sat_norm = ((df["era5_soil_moisture"].ffill() - soil_min) / (soil_max - soil_min + 1e-6)).clip(0.05, 1.0)
        tank_norm = (
            df["tank_168h"] / (df["tank_168h"].expanding(min_periods=48).quantile(0.95) + 1e-6)
        ).clip(0.05, 1.5)
        df["moisture_index"] = (0.5 * sat_norm + 0.5 * tank_norm).fillna(0.5)
        df["eff_rain_mm"] = df["fused_rain_wind_adj"] * (df["moisture_index"] ** 1.5)

        # 4. Depth lags & velocity
        for lag in [1, 2, 3, 4, 6, 12, 24]:
            df[f"ddepth_{lag}h"] = df["depth_mm"] - df["depth_mm"].shift(lag)
        df["accel_1h"] = df["ddepth_1h"] - df["ddepth_1h"].shift(1)

        # 5. Rain lags & rolling sums
        for lag in range(0, 7):
            df[f"fused_rain_lag_{lag}h"] = df["fused_rain_wind_adj"].shift(lag)
            df[f"ref_rain_lag_{lag}h"] = rain_ref.shift(lag)
            df[f"eff_rain_lag_{lag}h"] = df["eff_rain_mm"].shift(lag)

        for win in [2, 3, 6, 12, 24]:
            df[f"fused_rain_sum_{win}h"] = df["fused_rain_wind_adj"].rolling(win).sum()
            df[f"ref_rain_sum_{win}h"] = rain_ref.rolling(win).sum()
            df[f"eff_rain_sum_{win}h"] = df["eff_rain_mm"].rolling(win).sum()

        df["press_hpa"] = df["swcc_press_hpa"].fillna(df["stump_press_hpa"]).fillna(df["era5_press_hpa"])
        df["dpress_3h"] = df["press_hpa"] - df["press_hpa"].shift(3)
        df["dpress_6h"] = df["press_hpa"] - df["press_hpa"].shift(6)

        return df

    def _attach_horizon_future_cols(self, df: pd.DataFrame, h: int) -> pd.DataFrame:
        work = df.copy()
        work["fut_ref_tank_6h"] = work["ref_tank_6h"].shift(-h)
        work["fut_ref_tank_24h"] = work["tank_24h"].shift(-h)
        work["fut_ref_tank_96h"] = work["ref_tank_96h"].shift(-h)
        work["fut_rain_sum_6h_at_target"] = work["ref_rain_sum_6h"].shift(-h)
        work["fut_rain_sum_12h_at_target"] = work["ref_rain_sum_12h"].shift(-h)
        work["fut_eff_rain_sum_6h_at_target"] = work["eff_rain_sum_6h"].shift(-h)
        for k in range(0, min(h, 6)):
            work[f"fut_rain_minus_{k}h_from_target"] = work["rain_mm"].shift(-(h - k))
        return work

    def _train_and_save_models(self, df_feat: pd.DataFrame) -> None:
        logger.info(
            "Training multi-horizon P10/P50/P90 Hydro-GBDT models strictly on training split (< %s)...",
            TRAIN_SPLIT_UTC,
        )
        trained: dict[int, dict[str, HistGradientBoostingRegressor]] = {}

        modern_era_ts = pd.Timestamp("2024-05-17 00:00:00+00:00")
        for h in HORIZONS:
            work = self._attach_horizon_future_cols(df_feat, h)
            work["target_delta"] = work["depth_mm"].shift(-h) - work["depth_mm"]
            f_cols = CAUSAL_COLS + nwp_cols_for_horizon(h)

            # Enforce strict chronological split: only train where target arrival timestamp < TRAIN_SPLIT_UTC
            target_ts = work.index + pd.Timedelta(hours=h)
            train_slice = work[target_ts < TRAIN_SPLIT_UTC]
            valid = train_slice.dropna(
                subset=["target_delta", "depth_mm", "ddepth_1h", "ddepth_24h"] + f_cols
            ).copy()
            X = valid[f_cols]
            y = valid["target_delta"].values
            # Weight modern 2024-2025 multi-gauge era 2.5x so short horizons prioritize The Stump while 6-48h leverages 2015-2025
            sample_w = np.where(valid.index >= modern_era_ts, 2.5, 1.0)

            m_p50 = HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=300,
                learning_rate=0.04,
                max_depth=6,
                min_samples_leaf=25,
                l2_regularization=2.0,
                random_state=42,
            )
            m_p10 = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.10,
                max_iter=220,
                learning_rate=0.04,
                max_depth=5,
                min_samples_leaf=30,
                random_state=42,
            )
            m_p90 = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.90,
                max_iter=220,
                learning_rate=0.04,
                max_depth=5,
                min_samples_leaf=30,
                random_state=42,
            )
            m_p50.fit(X, y, sample_weight=sample_w)
            m_p10.fit(X, y, sample_weight=sample_w)
            m_p90.fit(X, y, sample_weight=sample_w)
            trained[h] = {"p10": m_p10, "p50": m_p50, "p90": m_p90}

        self.models = trained
        joblib.dump(self.models, self.models_path)
        logger.info("Saved trained models to %s", self.models_path)

    async def refresh_live_telemetry(self) -> dict[str, Any]:
        """Fetch latest ThingSpeak packets + Open-Meteo 48h NWP forecast and recompute dashboard state."""
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                # 1. Fetch recent ThingSpeak entries (last 400 packets ~ 2-3 days)
                ts_resp = await client.get(
                    "https://api.thingspeak.com/channels/2574933/feeds.json?results=400"
                )
                ts_resp.raise_for_status()
                feeds = ts_resp.json().get("feeds", [])

                if feeds:
                    df_new = pd.DataFrame(feeds)
                    df_new["timestamp"] = pd.to_datetime(df_new["created_at"], utc=True)
                    df_new["depth_mm"] = pd.to_numeric(df_new.get("field1"), errors="coerce")
                    df_new["swcc_rain_mm"] = pd.to_numeric(df_new.get("field2"), errors="coerce")
                    df_new["swcc_press_hpa"] = pd.to_numeric(df_new.get("field3"), errors="coerce")
                    df_new["stump_rain_mm"] = pd.to_numeric(df_new.get("field4"), errors="coerce")
                    df_new["stump_press_hpa"] = pd.to_numeric(df_new.get("field5"), errors="coerce")
                    df_new = df_new.sort_values("timestamp").set_index("timestamp")

                    new_1h = pd.DataFrame(
                        index=pd.date_range(
                            df_new.index.min().floor("1h"),
                            df_new.index.max().floor("1h"),
                            freq="1h",
                            tz="UTC",
                        )
                    )
                    new_1h.index.name = "timestamp"
                    new_1h["depth_mm_raw"] = df_new["depth_mm"].resample("1h").mean()
                    new_1h["depth_mm"] = new_1h["depth_mm_raw"].interpolate(method="time", limit=3)
                    new_1h["swcc_rain_mm"] = df_new["swcc_rain_mm"].resample("1h").sum(min_count=1)
                    new_1h["stump_rain_mm"] = df_new["stump_rain_mm"].resample("1h").sum(min_count=1)
                    new_1h["swcc_press_hpa"] = df_new["swcc_press_hpa"].resample("1h").mean()
                    new_1h["stump_press_hpa"] = df_new["stump_press_hpa"].resample("1h").mean()

                    # Merge into self.df_hourly
                    combined = self.df_hourly.combine_first(new_1h)
                    for c in [
                        "depth_mm_raw",
                        "depth_mm",
                        "swcc_rain_mm",
                        "stump_rain_mm",
                        "swcc_press_hpa",
                        "stump_press_hpa",
                    ]:
                        if c in new_1h.columns:
                            combined.loc[new_1h.index, c] = new_1h[c].combine_first(
                                combined.loc[new_1h.index, c]
                            )
                    self.df_hourly = combined.sort_index()

                # 2. Fetch Open-Meteo Live NWP Forecast (past 3 days + next 3 days)
                om_url = (
                    "https://api.open-meteo.com/v1/forecast"
                    "?latitude=51.8258&longitude=-3.6611"
                    "&hourly=precipitation,temperature_2m,wind_speed_10m,surface_pressure,"
                    "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,evapotranspiration"
                    "&past_days=3&forecast_days=3&timezone=UTC"
                )
                om_resp = await client.get(om_url)
                if om_resp.status_code == 200:
                    om = om_resp.json()["hourly"]
                    df_om = pd.DataFrame(
                        {
                            "timestamp": pd.to_datetime(om["time"], utc=True),
                            "rain_mm": om["precipitation"],
                            "temp_c": om["temperature_2m"],
                            "wind_kmh": om["wind_speed_10m"],
                            "era5_press_hpa": om["surface_pressure"],
                            "era5_soil_moisture": om["soil_moisture_0_to_7cm"],
                            "era5_soil_deep": om["soil_moisture_7_to_28cm"],
                            "et0_mm": om["evapotranspiration"],
                        }
                    ).set_index("timestamp")

                    # Update historical weather columns and store future weather rows
                    wx_cols = [
                        "rain_mm",
                        "temp_c",
                        "wind_kmh",
                        "era5_press_hpa",
                        "era5_soil_moisture",
                        "era5_soil_deep",
                        "et0_mm",
                    ]
                    common_idx = self.df_hourly.index.intersection(df_om.index)
                    for c in wx_cols:
                        self.df_hourly.loc[common_idx, c] = df_om.loc[common_idx, c].combine_first(
                            self.df_hourly.loc[common_idx, c]
                        )

                    last_obs_ts = self.df_hourly.dropna(subset=["depth_mm"]).index.max()
                    self.df_future_wx = df_om[df_om.index > last_obs_ts].head(48).copy()

                    # Persist updated hourly parquet to PVC
                    self.df_hourly.to_parquet(self.aligned_1h_path)

            self.last_sync_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            self.sync_status = "live_synced"
        except Exception as exc:
            logger.warning("Live telemetry sync fallback (using cached PVC data): %s", exc)
            self.sync_status = f"cached_offline ({type(exc).__name__})"
            if not self.last_sync_utc:
                self.last_sync_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        df_feat = self._engineer_features(self.df_hourly)
        self.latest_payload = self._build_dashboard_payload(df_feat, self.df_future_wx)
        self._ensure_and_update_verification_ledger(df_feat)
        return self.latest_payload

    def _build_dashboard_payload(
        self, df_feat: pd.DataFrame, df_future_wx: pd.DataFrame
    ) -> dict[str, Any]:
        valid_obs = df_feat.dropna(subset=["depth_mm"]).sort_index()
        now_ts = valid_obs.index.max()
        now_row = valid_obs.loc[now_ts]

        # Build extended DataFrame (observed + 48h future weather) so future tank states roll forward
        if df_future_wx is not None and not df_future_wx.empty:
            fut_wx = df_future_wx.copy()
        else:
            # Fallback if offline: project zero rain or recent climatology over next 48h
            fut_idx = pd.date_range(now_ts + pd.Timedelta(hours=1), periods=48, freq="1h", tz="UTC")
            fut_wx = pd.DataFrame(
                {
                    "rain_mm": 0.0,
                    "temp_c": float(now_row.get("temp_c", 9.0)),
                    "wind_kmh": float(now_row.get("wind_kmh", 15.0)),
                    "era5_press_hpa": float(now_row.get("press_hpa", 985.0)),
                    "era5_soil_moisture": float(now_row.get("era5_soil_moisture", 0.39)),
                    "era5_soil_deep": float(now_row.get("era5_soil_deep", 0.39)),
                    "et0_mm": 0.05,
                },
                index=fut_idx,
            )
            fut_wx.index.name = "timestamp"

        ext_df = pd.concat([valid_obs.tail(800), fut_wx], axis=0)
        ext_df = ext_df[~ext_df.index.duplicated(keep="first")].sort_index()
        ext_feat = self._engineer_features(ext_df)

        # Predict at anchor timestamp `now_ts` across all HORIZONS
        base_depth = float(now_row["depth_mm"])
        baseflow_30d = float(now_row["baseflow_30d"])
        anchor_Points = [
            {
                "horizon_h": 0,
                "timestamp": now_ts.isoformat(),
                "p10_mm": round(base_depth, 1),
                "p50_mm": round(base_depth, 1),
                "p90_mm": round(base_depth, 1),
                "rain_mm": round(float(now_row.get("fused_rain_mm", 0.0)), 2),
                "moisture_index": round(float(now_row.get("moisture_index", 0.5)), 3),
            }
        ]

        for h in HORIZONS:
            work_h = self._attach_horizon_future_cols(ext_feat, h)
            f_cols = CAUSAL_COLS + nwp_cols_for_horizon(h)
            x_row = work_h.loc[[now_ts], f_cols]

            d_p10 = float(self.models[h]["p10"].predict(x_row)[0])
            d_p50 = float(self.models[h]["p50"].predict(x_row)[0])
            d_p90 = float(self.models[h]["p90"].predict(x_row)[0])

            p10 = max(350.0, base_depth + min(d_p10, d_p50))
            p50 = max(350.0, base_depth + d_p50)
            p90 = max(p50, base_depth + max(d_p90, d_p50))

            target_ts = now_ts + pd.Timedelta(hours=h)
            fut_rain = (
                float(ext_feat.loc[target_ts, "rain_mm"])
                if target_ts in ext_feat.index and pd.notna(ext_feat.loc[target_ts, "rain_mm"])
                else 0.0
            )
            fut_moist = (
                float(ext_feat.loc[target_ts, "moisture_index"])
                if target_ts in ext_feat.index
                else float(now_row.get("moisture_index", 0.5))
            )

            anchor_Points.append(
                {
                    "horizon_h": h,
                    "timestamp": target_ts.isoformat(),
                    "p10_mm": round(p10, 1),
                    "p50_mm": round(p50, 1),
                    "p90_mm": round(p90, 1),
                    "rain_mm": round(fut_rain, 2),
                    "moisture_index": round(fut_moist, 3),
                }
            )

        # Interpolate smooth hourly trajectory from 0..48h for the hydrograph & table
        df_anchors = pd.DataFrame(anchor_Points).set_index("horizon_h")
        full_h = pd.Index(range(0, 49), name="horizon_h")
        df_interp = df_anchors[["p10_mm", "p50_mm", "p90_mm"]].reindex(full_h).interpolate("index")

        forecast_hourly = []
        for h_step in range(1, 49):
            ts_step = now_ts + pd.Timedelta(hours=h_step)
            r_val = (
                float(ext_feat.loc[ts_step, "rain_mm"])
                if ts_step in ext_feat.index and pd.notna(ext_feat.loc[ts_step, "rain_mm"])
                else 0.0
            )
            m_val = (
                float(ext_feat.loc[ts_step, "moisture_index"])
                if ts_step in ext_feat.index and pd.notna(ext_feat.loc[ts_step, "moisture_index"])
                else float(now_row.get("moisture_index", 0.5))
            )
            p50_v = round(float(df_interp.loc[h_step, "p50_mm"]), 1)
            p90_v = round(float(df_interp.loc[h_step, "p90_mm"]), 1)
            p10_v = round(float(df_interp.loc[h_step, "p10_mm"]), 1)
            status = (
                "RED"
                if p50_v >= RED_DEPTH_MM
                else ("AMBER" if p90_v >= RED_DEPTH_MM or p50_v >= AMBER_DEPTH_MM else "GREEN")
            )
            forecast_hourly.append(
                {
                    "horizon_h": h_step,
                    "timestamp": ts_step.isoformat(),
                    "p10_mm": p10_v,
                    "p50_mm": p50_v,
                    "p90_mm": p90_v,
                    "step_inches_p50": round((p50_v - STEP_DATUM_MM) / 25.4, 1),
                    "rain_mm": round(r_val, 2),
                    "moisture_index": round(m_val, 3),
                    "status": status,
                }
            )

        # Past 72h observed history
        hist_slice = valid_obs.tail(72)
        history_hourly = [
            {
                "timestamp": ts.isoformat(),
                "depth_mm": round(float(r["depth_mm"]), 1),
                "step_inches": round((float(r["depth_mm"]) - STEP_DATUM_MM) / 25.4, 1),
                "fused_rain_mm": round(float(r.get("fused_rain_mm", 0.0)), 2),
                "swcc_rain_mm": (
                    round(float(r["swcc_rain_mm"]), 2) if pd.notna(r.get("swcc_rain_mm")) else None
                ),
                "stump_rain_mm": (
                    round(float(r["stump_rain_clean"]), 2)
                    if pd.notna(r.get("stump_rain_clean"))
                    else None
                ),
                "moisture_index": round(float(r.get("moisture_index", 0.5)), 3),
            }
            for ts, r in hist_slice.iterrows()
        ]

        # Safety Banner & Clearance / Flood Countdown logic
        max_p50_6h = max(p["p50_mm"] for p in forecast_hourly[:6])
        max_p90_12h = max(p["p90_mm"] for p in forecast_hourly[:12])
        max_p50_24h = max(p["p50_mm"] for p in forecast_hourly[:24])
        max_p90_24h = max(p["p90_mm"] for p in forecast_hourly[:24])

        if base_depth >= RED_DEPTH_MM or max_p50_6h >= RED_DEPTH_MM:
            overall_alert = "RED"
            alert_headline = "FLOOD PULSE ACTIVE OR IMMINENT (≥700 mm / >6\" over The Step)"
        elif (
            base_depth >= AMBER_DEPTH_MM
            or max_p90_12h >= RED_DEPTH_MM
            or max_p50_24h >= RED_DEPTH_MM
        ):
            overall_alert = "AMBER"
            alert_headline = "CAUTION: ELEVATED WATER OR RISING FLOOD RISK IN NEXT 12–24H"
        else:
            overall_alert = "GREEN"
            alert_headline = "NORMAL STREAMWAY CONDITIONS (<620 mm, P90 <700 mm)"

        # Countdown calculation
        if base_depth >= RED_DEPTH_MM:
            clear_p50 = next(
                (p["horizon_h"] for p in forecast_hourly if p["p50_mm"] < RED_DEPTH_MM), None
            )
            clear_p90 = next(
                (p["horizon_h"] for p in forecast_hourly if p["p90_mm"] < RED_DEPTH_MM), None
            )
            countdown_label = "Estimated Recession Below 700 mm"
            countdown_value = (
                f"+{clear_p50}h (P50) / +{clear_p90}h (P90)"
                if clear_p50 and clear_p90
                else (f"+{clear_p50}h (P50)" if clear_p50 else ">48h")
            )
        else:
            cross_p90 = next(
                (p["horizon_h"] for p in forecast_hourly if p["p90_mm"] >= RED_DEPTH_MM), None
            )
            cross_p50 = next(
                (p["horizon_h"] for p in forecast_hourly if p["p50_mm"] >= RED_DEPTH_MM), None
            )
            countdown_label = "Next ≥700 mm Flood Threshold Crossing"
            if cross_p50:
                countdown_value = f"In +{cross_p50}h (P50 Median) / +{cross_p90}h (P90 Upper)"
            elif cross_p90:
                countdown_value = f"P90 Risk in +{cross_p90}h (P50 stays <700 mm)"
            else:
                countdown_value = "None in next 48h"

        # Gauge health check
        last_swcc_ts = valid_obs.dropna(subset=["swcc_rain_mm"]).index.max()
        last_stump_ts = valid_obs.dropna(subset=["stump_rain_clean"]).index.max()
        swcc_age_h = (now_ts - last_swcc_ts).total_seconds() / 3600.0 if pd.notna(last_swcc_ts) else 999.0
        stump_age_h = (now_ts - last_stump_ts).total_seconds() / 3600.0 if pd.notna(last_stump_ts) else 999.0

        ddepth_1h = float(now_row.get("ddepth_1h", 0.0))
        if ddepth_1h >= 15.0:
            hydro_regime = "Rising Flood Limb"
        elif ddepth_1h <= -3.0:
            hydro_regime = "Recession Limb (Falling)"
        else:
            hydro_regime = "Quiescent / Baseflow"

        return {
            "generated_at_utc": self.last_sync_utc
            or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "sync_status": self.sync_status,
            "current": {
                "timestamp": now_ts.isoformat(),
                "depth_mm": round(base_depth, 1),
                "step_over_mm": round(base_depth - STEP_DATUM_MM, 1),
                "step_over_inches": round((base_depth - STEP_DATUM_MM) / 25.4, 1),
                "baseflow_30d_mm": round(baseflow_30d, 1),
                "ddepth_1h_mm": round(ddepth_1h, 1),
                "ddepth_3h_mm": round(float(now_row.get("ddepth_3h", 0.0)), 1),
                "hydro_regime": hydro_regime,
                "moisture_index": round(float(now_row.get("moisture_index", 0.5)), 3),
                "moisture_pct": round(float(now_row.get("moisture_index", 0.5)) * 100.0, 1),
                "soil_era5": round(float(now_row.get("era5_soil_moisture", 0.38)), 3),
                "rain_6h_mm": round(float(now_row.get("fused_rain_sum_6h", 0.0)), 1),
                "rain_24h_mm": round(float(now_row.get("fused_rain_sum_24h", 0.0)), 1),
                "wind_kmh": round(float(now_row.get("wind_kmh", 0.0)), 1),
                "temp_c": round(float(now_row.get("temp_c", 0.0)), 1),
                "press_hpa": round(float(now_row.get("press_hpa", 985.0)), 1),
                "alert_level": overall_alert,
                "alert_headline": alert_headline,
                "countdown_label": countdown_label,
                "countdown_value": countdown_value,
                "max_p50_24h_mm": round(max_p50_24h, 1),
                "max_p90_24h_mm": round(max_p90_24h, 1),
            },
            "gauges": {
                "depth_sensor": {
                    "name": "OFD1 Streamway Pressure Sensor (field1)",
                    "status": "ONLINE",
                    "cadence": "10 min",
                    "last_seen": now_ts.strftime("%Y-%m-%d %H:%M UTC"),
                },
                "stump_gauge": {
                    "name": "The Stump Upland Rain/Press (field4/5)",
                    "status": "ONLINE" if stump_age_h <= 4.0 else "STALE / FAILOVER",
                    "cadence": "30 min",
                    "last_seen": last_stump_ts.strftime("%Y-%m-%d %H:%M UTC")
                    if pd.notna(last_stump_ts)
                    else "N/A",
                    "glitch_filter": "Active (Oct 2026 96.6mm/h latch masked)",
                },
                "swcc_gauge": {
                    "name": "SWCC HQ Rain/Press (field2/3)",
                    "status": "ONLINE" if swcc_age_h <= 4.0 else "OFFLINE (Auto-Failover Active)",
                    "cadence": "30 min",
                    "last_seen": last_swcc_ts.strftime("%Y-%m-%d %H:%M UTC")
                    if pd.notna(last_swcc_ts)
                    else "N/A",
                },
            },
            "history_72h": history_hourly,
            "forecast_48h": forecast_hourly,
            "forecast_anchors": anchor_Points,
        }

    def simulate_what_if(
        self, rain_6h_mm: float, moisture_pct: float, start_depth_mm: float | None = None
    ) -> dict[str, Any]:
        """Simulate the next 24h streamway response to a synthetic 6-hour rainstorm at specified peat saturation."""
        valid_obs = self.df_hourly.dropna(subset=["depth_mm"]).sort_index().tail(750).copy()
        now_ts = valid_obs.index.max()

        if start_depth_mm is not None:
            valid_obs.loc[now_ts, "depth_mm"] = float(start_depth_mm)

        # Create 24h future scenario: rain falls uniformly over hours +1..+6, then 0 for +7..+24
        hourly_rain = float(rain_6h_mm) / 6.0
        fut_idx = pd.date_range(now_ts + pd.Timedelta(hours=1), periods=24, freq="1h", tz="UTC")
        moist_frac = np.clip(moisture_pct / 100.0, 0.05, 1.30)
        soil_sim = 0.31 + 0.11 * min(moist_frac, 1.0)

        fut_wx = pd.DataFrame(
            {
                "rain_mm": [hourly_rain if i < 6 else 0.0 for i in range(24)],
                "swcc_rain_mm": [hourly_rain if i < 6 else 0.0 for i in range(24)],
                "stump_rain_mm": [hourly_rain if i < 6 else 0.0 for i in range(24)],
                "temp_c": 9.0,
                "wind_kmh": 22.0,
                "era5_press_hpa": 980.0,
                "era5_soil_moisture": soil_sim,
                "era5_soil_deep": soil_sim,
                "et0_mm": 0.03,
            },
            index=fut_idx,
        )
        fut_wx.index.name = "timestamp"

        sim_df = pd.concat([valid_obs, fut_wx], axis=0)
        sim_feat = self._engineer_features(sim_df)
        # Override moisture index and slow tanks around now_ts..end to match requested peat saturation
        sim_feat.loc[sim_feat.index >= now_ts - pd.Timedelta(hours=6), "moisture_index"] = moist_frac
        sim_feat.loc[sim_feat.index >= now_ts - pd.Timedelta(hours=6), "era5_soil_moisture"] = soil_sim
        sim_feat.loc[sim_feat.index >= now_ts - pd.Timedelta(hours=6), "tank_96h"] = 0.55 * moist_frac
        sim_feat.loc[sim_feat.index >= now_ts - pd.Timedelta(hours=6), "tank_168h"] = 0.55 * moist_frac
        sim_feat.loc[sim_feat.index >= now_ts - pd.Timedelta(hours=6), "ref_tank_96h"] = 0.55 * moist_frac
        sim_feat["eff_rain_mm"] = sim_feat["fused_rain_wind_adj"] * (sim_feat["moisture_index"] ** 1.5)
        for win in [2, 3, 6, 12, 24]:
            sim_feat[f"eff_rain_sum_{win}h"] = sim_feat["eff_rain_mm"].rolling(win).sum()

        # Since rain falls in +1..+6h and Tc = 2h, evaluate model response as the storm unfolds
        base_d = float(sim_feat.loc[now_ts, "depth_mm"])
        sim_points = [
            {
                "hour": 0,
                "p10_mm": round(base_d, 1),
                "p50_mm": round(base_d, 1),
                "p90_mm": round(base_d, 1),
                "rain_mm": round(hourly_rain, 2),
            }
        ]
        for h in [1, 2, 3, 6, 12, 18, 24]:
            work_h = self._attach_horizon_future_cols(sim_feat, h)
            f_cols = CAUSAL_COLS + nwp_cols_for_horizon(h)
            x_row = work_h.loc[[now_ts], f_cols].copy()
            # Scale raw rain inputs by antecedent peat runoff efficiency (sponge effect when dry vs conduit when wet)
            runoff_eff = float(np.clip((moist_frac / 0.75) ** 1.6, 0.12, 1.45))
            for col in x_row.columns:
                if "rain" in col or "tank_6h" in col or "tank_24h" in col:
                    x_row[col] = x_row[col] * runoff_eff
            d50 = float(self.models[h]["p50"].predict(x_row)[0]) * runoff_eff
            d10 = float(self.models[h]["p10"].predict(x_row)[0]) * runoff_eff
            d90 = float(self.models[h]["p90"].predict(x_row)[0]) * runoff_eff
            peat_boost = max(
                0.0, (rain_6h_mm * (moist_frac**1.8) * 18.0) * np.exp(-((h - 6.0) ** 2) / 30.0)
            )
            p50 = max(380.0, base_d + d50 + 0.35 * peat_boost)
            p10 = max(360.0, base_d + min(d10, d50) + 0.20 * peat_boost)
            p90 = max(p50 + 15.0, base_d + max(d90, d50) + 0.55 * peat_boost)
            sim_points.append(
                {
                    "hour": h,
                    "p10_mm": round(p10, 1),
                    "p50_mm": round(p50, 1),
                    "p90_mm": round(p90, 1),
                    "rain_mm": round(hourly_rain if h <= 6 else 0.0, 2),
                }
            )

        df_sim = pd.DataFrame(sim_points).set_index("hour")
        df_sim_interp = df_sim[["p10_mm", "p50_mm", "p90_mm"]].reindex(range(0, 25)).interpolate("index")
        curve = [
            {
                "hour": int(hr),
                "p10_mm": round(float(df_sim_interp.loc[hr, "p10_mm"]), 1),
                "p50_mm": round(float(df_sim_interp.loc[hr, "p50_mm"]), 1),
                "p90_mm": round(float(df_sim_interp.loc[hr, "p90_mm"]), 1),
                "rain_mm": round(hourly_rain if 1 <= hr <= 6 else 0.0, 2),
            }
            for hr in range(0, 25)
        ]
        peak_p50 = max(c["p50_mm"] for c in curve)
        peak_p90 = max(c["p90_mm"] for c in curve)
        return {
            "rain_6h_mm": rain_6h_mm,
            "moisture_pct": moisture_pct,
            "start_depth_mm": round(base_d, 1),
            "peak_p50_mm": round(peak_p50, 1),
            "peak_p90_mm": round(peak_p90, 1),
            "net_rise_p50_mm": round(peak_p50 - base_d, 1),
            "crosses_700mm_p50": bool(peak_p50 >= RED_DEPTH_MM),
            "crosses_700mm_p90": bool(peak_p90 >= RED_DEPTH_MM),
            "trajectory_24h": curve,
        }

    def get_diagnostics_summary(self) -> dict[str, Any]:
        """Return Phase 1 & Phase 2 diagnostics (CCF, ACF, Dry Rain, and 1h-48h Horizon Breakdown)."""
        df = self._engineer_features(self.df_hourly)
        df["delta_1h"] = df["depth_mm"].diff(1)

        lags = list(range(0, 15))
        ccf_stump = [round(float(df["stump_rain_clean"].corr(df["delta_1h"].shift(-k))), 3) for k in lags]
        ccf_swcc = [round(float(df["swcc_rain_mm"].corr(df["delta_1h"].shift(-k))), 3) for k in lags]
        ccf_ref = [round(float(df["rain_mm"].corr(df["delta_1h"].shift(-k))), 3) for k in lags]

        rising_mask = df["delta_1h"] > 15.0
        recession_mask = (df["delta_1h"] < -2.0) & (df["rain_mm"] == 0)
        acf_overall = [round(float(df["depth_mm"].autocorr(lag=k)), 3) for k in lags]
        acf_rising = [
            round(float(df.loc[rising_mask, "depth_mm"].corr(df["depth_mm"].shift(-k).loc[rising_mask])), 3)
            for k in lags
        ]
        acf_recession = [
            round(
                float(df.loc[recession_mask, "depth_mm"].corr(df["depth_mm"].shift(-k).loc[recession_mask])),
                3,
            )
            for k in lags
        ]

        # Dry Rain storm events
        df["swcc_6h"] = df["swcc_rain_mm"].rolling(6).sum()
        df["peak_depth_6h"] = df["depth_mm"].rolling(8).max().shift(-4)
        is_max = df["swcc_6h"] == df["swcc_6h"].rolling(13, center=True).max()
        storms = df[is_max & (df["swcc_6h"] >= 6.0)].dropna(
            subset=["peak_depth_6h", "moisture_index", "swcc_6h"]
        )
        storms_list = [
            {
                "timestamp": ts.strftime("%Y-%m-%d %H:%M"),
                "rain_6h_mm": round(float(r["swcc_6h"]), 1),
                "effective_rain": round(float(r["swcc_6h"] * (r["moisture_index"] ** 1.5)), 1),
                "peak_depth_mm": round(float(r["peak_depth_6h"]), 0),
                "moisture_index": round(float(r["moisture_index"]), 2),
            }
            for ts, r in storms.iterrows()
        ]

        breakdown = (
            pd.read_parquet(self.breakdown_path).to_dict(orient="records")
            if self.breakdown_path.exists()
            else []
        )
        ablation = (
            pd.read_parquet(self.metrics_path).to_dict(orient="records")
            if self.metrics_path.exists()
            else []
        )

        return {
            "lags": lags,
            "ccf": {"stump": ccf_stump, "swcc": ccf_swcc, "reference": ccf_ref},
            "acf": {"overall": acf_overall, "rising": acf_rising, "recession": acf_recession},
            "dry_rain_storms": storms_list,
            "horizon_breakdown": breakdown,
            "ablation_metrics": ablation,
        }

    def _ensure_and_update_verification_ledger(self, df_feat: pd.DataFrame) -> None:
        """Seed full 2015-2026 verification ledger (tagged train vs out-of-sample test), append live forecasts, and reconcile actual depths."""
        ledger_horizons = [1, 2, 3, 6, 12, 24]
        valid_obs = df_feat.dropna(subset=["depth_mm"]).sort_index()

        need_seed = not self.ledger_path.exists()
        if not need_seed:
            _existing = pd.read_parquet(self.ledger_path)
            if "train_in_sample" not in _existing["source"].unique():
                need_seed = True
            elif pd.to_datetime(_existing["target_timestamp"].min(), utc=True) > valid_obs.index.min() + pd.Timedelta(days=30):
                need_seed = True

        if need_seed:
            logger.info(
                "Building full 2015–2026 verification ledger (Train < %s vs Out-of-Sample Test >= %s)...",
                TRAIN_SPLIT_UTC,
                TRAIN_SPLIT_UTC,
            )
            ledger_chunks = []

            for h in ledger_horizons:
                work_h = self._attach_horizon_future_cols(df_feat, h)
                f_cols = CAUSAL_COLS + nwp_cols_for_horizon(h)
                sub = work_h.dropna(subset=["depth_mm", "ddepth_1h", "ddepth_24h"] + f_cols).copy()
                if sub.empty:
                    continue

                X_sub = sub[f_cols]
                base_d = sub["depth_mm"].values
                d10 = self.models[h]["p10"].predict(X_sub)
                d50 = self.models[h]["p50"].predict(X_sub)
                d90 = self.models[h]["p90"].predict(X_sub)

                p50 = np.maximum(350.0, base_d + d50)
                p10 = np.maximum(350.0, base_d + np.minimum(d10, d50))
                p90 = np.maximum(p50, base_d + np.maximum(d90, d50))

                issued_idx = sub.index
                target_idx = issued_idx + pd.Timedelta(hours=h)
                sources = np.where(
                    target_idx < TRAIN_SPLIT_UTC, "train_in_sample", "test_out_of_sample"
                )

                chunk = pd.DataFrame(
                    {
                        "issued_at": issued_idx,
                        "target_timestamp": target_idx,
                        "horizon_h": h,
                        "start_depth_mm": np.round(base_d, 1),
                        "pred_p10_mm": np.round(p10, 1),
                        "pred_p50_mm": np.round(p50, 1),
                        "pred_p90_mm": np.round(p90, 1),
                        "actual_depth_mm": np.nan,
                        "rain_mm": np.nan,
                        "source": sources,
                    }
                )
                ledger_chunks.append(chunk)

            df_ledger = pd.concat(ledger_chunks, ignore_index=True)
        else:
            df_ledger = pd.read_parquet(self.ledger_path)

        # Append current live forecast points from self.latest_payload
        if self.latest_payload and "forecast_anchors" in self.latest_payload:
            now_str = self.latest_payload["current"]["timestamp"]
            issued_ts = pd.to_datetime(now_str, utc=True)
            base_now = float(self.latest_payload["current"]["depth_mm"])
            live_rows = []
            for pt in self.latest_payload["forecast_anchors"]:
                h_val = int(pt["horizon_h"])
                if h_val in ledger_horizons:
                    live_rows.append(
                        {
                            "issued_at": issued_ts,
                            "target_timestamp": pd.to_datetime(pt["timestamp"], utc=True),
                            "horizon_h": h_val,
                            "start_depth_mm": round(base_now, 1),
                            "pred_p10_mm": float(pt["p10_mm"]),
                            "pred_p50_mm": float(pt["p50_mm"]),
                            "pred_p90_mm": float(pt["p90_mm"]),
                            "actual_depth_mm": np.nan,
                            "rain_mm": float(pt.get("rain_mm", 0.0)),
                            "source": "live_forecast",
                        }
                    )
            if live_rows:
                df_live = pd.DataFrame(live_rows)
                df_ledger = pd.concat([df_ledger, df_live], ignore_index=True)
                df_ledger = df_ledger.drop_duplicates(
                    subset=["target_timestamp", "horizon_h"], keep="last"
                )

        # Reconcile actual measured depth & rainfall from valid_obs where target_timestamp has arrived
        depth_map = valid_obs["depth_mm"].round(1)
        rain_map = valid_obs["fused_rain_mm"].round(2)
        matched_depths = df_ledger["target_timestamp"].map(depth_map)
        matched_rains = df_ledger["target_timestamp"].map(rain_map)
        df_ledger["actual_depth_mm"] = matched_depths.combine_first(df_ledger["actual_depth_mm"])
        df_ledger["rain_mm"] = matched_rains.combine_first(df_ledger["rain_mm"]).fillna(0.0)

        df_ledger = df_ledger.sort_values(["target_timestamp", "horizon_h"]).reset_index(drop=True)
        df_ledger.to_parquet(self.ledger_path, index=False)

    def get_verification_history(
        self, horizon_h: int = 2, window: str = "recent_14d"
    ) -> dict[str, Any]:
        """Return historical forecast vs. actual verification metrics and time series."""
        if not self.ledger_path.exists():
            df_feat = self._engineer_features(self.df_hourly)
            self._ensure_and_update_verification_ledger(df_feat)

        df_ledger = pd.read_parquet(self.ledger_path)
        df_ledger["target_timestamp"] = pd.to_datetime(df_ledger["target_timestamp"], utc=True)
        df_ledger["issued_at"] = pd.to_datetime(df_ledger["issued_at"], utc=True)

        max_obs_ts = df_ledger.dropna(subset=["actual_depth_mm"])["target_timestamp"].max()
        min_obs_ts = df_ledger.dropna(subset=["actual_depth_mm"])["target_timestamp"].min()
        window_ranges = {
            "recent_14d": (max_obs_ts - pd.Timedelta(days=14), max_obs_ts + pd.Timedelta(hours=24)),
            "oct_2026_floods": (
                pd.Timestamp("2026-10-01T00:00:00Z"),
                pd.Timestamp("2026-10-25T23:59:59Z"),
            ),
            "aug_2026_dry_rain": (
                pd.Timestamp("2026-08-10T00:00:00Z"),
                pd.Timestamp("2026-09-05T23:59:59Z"),
            ),
            "jan_feb_2026_winter": (
                pd.Timestamp("2026-01-15T00:00:00Z"),
                pd.Timestamp("2026-02-15T23:59:59Z"),
            ),
            "full_2026": (
                pd.Timestamp("2026-01-01T00:00:00Z"),
                max_obs_ts + pd.Timedelta(hours=24),
            ),
            "train_2024_2025": (
                min_obs_ts,
                pd.Timestamp("2025-12-31T23:59:59Z"),
            ),
            "full_all_2024_2026": (
                min_obs_ts,
                max_obs_ts + pd.Timedelta(hours=24),
            ),
        }
        start_ts, end_ts = window_ranges.get(window, window_ranges["oct_2026_floods"])

        # Compute Train (2024-2025) vs Out-of-Sample Test (2026) transparency metadata for selected horizon_h
        h_all = df_ledger[df_ledger["horizon_h"] == horizon_h].dropna(subset=["actual_depth_mm"])
        h_train = h_all[h_all["target_timestamp"] < TRAIN_SPLIT_UTC]
        h_test = h_all[h_all["target_timestamp"] >= TRAIN_SPLIT_UTC]

        def _quick_stats(sub_df: pd.DataFrame) -> dict[str, Any]:
            if sub_df.empty:
                return {"hours": 0, "mae_mm": 0.0, "rmse_mm": 0.0, "nse": 0.0, "cov_pct": 0.0}
            e = sub_df["pred_p50_mm"] - sub_df["actual_depth_mm"]
            mae_v = float(np.abs(e).mean())
            rmse_v = float(np.sqrt((e**2).mean()))
            vy = float(np.var(sub_df["actual_depth_mm"]))
            nse_v = 1.0 - (float((e**2).mean()) / vy) if vy > 0 else 0.0
            cov_v = (
                (sub_df["actual_depth_mm"] >= sub_df["pred_p10_mm"])
                & (sub_df["actual_depth_mm"] <= sub_df["pred_p90_mm"])
            ).mean() * 100.0
            return {
                "hours": int(len(sub_df)),
                "start_utc": sub_df["target_timestamp"].min().strftime("%Y-%m-%d %H:%M UTC"),
                "end_utc": sub_df["target_timestamp"].max().strftime("%Y-%m-%d %H:%M UTC"),
                "mae_mm": round(mae_v, 1),
                "rmse_mm": round(rmse_v, 1),
                "nse": round(nse_v, 3),
                "cov_pct": round(float(cov_v), 1),
            }

        valid_obs = self.df_hourly.dropna(subset=["depth_mm"]).sort_index()
        swcc_obs = self.df_hourly.dropna(subset=["swcc_rain_mm"]).sort_index()
        stump_obs = self.df_hourly.dropna(subset=["stump_rain_mm"]).sort_index()

        data_split_metadata = {
            "split_cutoff_iso": TRAIN_SPLIT_UTC.isoformat(),
            "split_cutoff_label": "2026-01-01 00:00 UTC",
            "train_split": _quick_stats(h_train),
            "test_split": _quick_stats(h_test),
            "sensor_spans": [
                {
                    "stream": "OFD1 Streamway Depth (field1)",
                    "start_utc": valid_obs.index.min().strftime("%Y-%m-%d %H:%M UTC"),
                    "end_utc": valid_obs.index.max().strftime("%Y-%m-%d %H:%M UTC"),
                    "hours": int(len(valid_obs)),
                    "role": "Target + Autoregressive Lags",
                },
                {
                    "stream": "SWCC HQ Rain & Pressure (field2/3)",
                    "start_utc": swcc_obs.index.min().strftime("%Y-%m-%d %H:%M UTC"),
                    "end_utc": swcc_obs.index.max().strftime("%Y-%m-%d %H:%M UTC"),
                    "hours": int(len(swcc_obs)),
                    "role": "Primary Valley Gauge (Offline Nov 4, 2026)",
                },
                {
                    "stream": "The Stump Upland Rain & Pressure (field4/5)",
                    "start_utc": stump_obs.index.min().strftime("%Y-%m-%d %H:%M UTC"),
                    "end_utc": stump_obs.index.max().strftime("%Y-%m-%d %H:%M UTC"),
                    "hours": int(len(stump_obs)),
                    "role": "Plateau Swallet Gauge (Oct 2–6 Glitch Masked)",
                },
                {
                    "stream": "Fan Gyhirych / Dan-yr-Ogof Reanalysis & NWP",
                    "start_utc": valid_obs.index.min().strftime("%Y-%m-%d %H:%M UTC"),
                    "end_utc": valid_obs.index.max().strftime("%Y-%m-%d %H:%M UTC"),
                    "hours": int(len(valid_obs)),
                    "role": "Peat Bog Tank + Wind + 48h NWP Forecast",
                },
            ],
        }

        # Compute summary scorecard across all horizons for the selected window
        horizon_scorecard = []
        for h_iter in [1, 2, 3, 6, 12, 24]:
            sub_h = df_ledger[
                (df_ledger["horizon_h"] == h_iter)
                & (df_ledger["target_timestamp"] >= start_ts)
                & (df_ledger["target_timestamp"] <= end_ts)
            ].dropna(subset=["actual_depth_mm"])
            if sub_h.empty:
                continue
            err = sub_h["pred_p50_mm"] - sub_h["actual_depth_mm"]
            pers_err = sub_h["start_depth_mm"] - sub_h["actual_depth_mm"]
            mae = float(np.abs(err).mean())
            rmse = float(np.sqrt((err**2).mean()))
            mse_pers = float((pers_err**2).mean())
            skill_pers = (1.0 - (float((err**2).mean()) / mse_pers)) * 100.0 if mse_pers > 0 else 0.0
            var_y = float(np.var(sub_h["actual_depth_mm"]))
            nse = 1.0 - (float((err**2).mean()) / var_y) if var_y > 0 else 0.0
            in_band = (
                (sub_h["actual_depth_mm"] >= sub_h["pred_p10_mm"])
                & (sub_h["actual_depth_mm"] <= sub_h["pred_p90_mm"])
            ).mean() * 100.0

            true_700 = sub_h["actual_depth_mm"] >= RED_DEPTH_MM
            pred_p50_700 = sub_h["pred_p50_mm"] >= RED_DEPTH_MM
            pred_p90_700 = sub_h["pred_p90_mm"] >= RED_DEPTH_MM
            tp_p50 = int((true_700 & pred_p50_700).sum())
            fp_p50 = int(((~true_700) & pred_p50_700).sum())
            fn_p50 = int((true_700 & (~pred_p50_700)).sum())
            tp_p90 = int((true_700 & pred_p90_700).sum())
            rec_p50 = (tp_p50 / int(true_700.sum()) * 100.0) if int(true_700.sum()) > 0 else 100.0
            rec_p90 = (tp_p90 / int(true_700.sum()) * 100.0) if int(true_700.sum()) > 0 else 100.0

            horizon_scorecard.append(
                {
                    "horizon_h": h_iter,
                    "verified_hours": len(sub_h),
                    "mae_mm": round(mae, 1),
                    "mae_inches": round(mae / 25.4, 2),
                    "rmse_mm": round(rmse, 1),
                    "nse": round(nse, 3),
                    "skill_vs_pers_pct": round(skill_pers, 1),
                    "band_coverage_pct": round(float(in_band), 1),
                    "flood_hours_actual": int(true_700.sum()),
                    "hits_p50": tp_p50,
                    "false_alarms_p50": fp_p50,
                    "missed_p50": fn_p50,
                    "recall_p50_pct": round(rec_p50, 1),
                    "recall_p90_pct": round(rec_p90, 1),
                }
            )

        # Extract time series for requested horizon_h
        sel_df = df_ledger[
            (df_ledger["horizon_h"] == horizon_h)
            & (df_ledger["target_timestamp"] >= start_ts)
            & (df_ledger["target_timestamp"] <= end_ts)
        ].copy()

        # Downsample large multi-month/multi-year windows so browser rendering stays snappy while keeping exact metrics
        stride = 24 if len(sel_df) > 15000 else (6 if len(sel_df) > 4000 else (3 if len(sel_df) > 1500 else 1))
        plot_df = sel_df.iloc[::stride].copy()

        series = []
        for _, r in plot_df.iterrows():
            act = float(r["actual_depth_mm"]) if pd.notna(r["actual_depth_mm"]) else None
            p50 = float(r["pred_p50_mm"])
            p10 = float(r["pred_p10_mm"])
            p90 = float(r["pred_p90_mm"])
            err_v = round(p50 - act, 1) if act is not None else None
            in_b = bool(p10 <= act <= p90) if act is not None else None
            series.append(
                {
                    "issued_at": r["issued_at"].isoformat(),
                    "target_timestamp": r["target_timestamp"].isoformat(),
                    "start_depth_mm": float(r["start_depth_mm"]),
                    "pred_p10_mm": p10,
                    "pred_p50_mm": p50,
                    "pred_p90_mm": p90,
                    "actual_depth_mm": act,
                    "error_mm": err_v,
                    "in_band": in_b,
                    "rain_mm": float(r.get("rain_mm", 0.0)),
                    "source": str(r.get("source", "2026_backtest")),
                }
            )

        # Latest 40 verified rows for the detailed audit table
        verified_tail = sel_df.dropna(subset=["actual_depth_mm"]).tail(40).iloc[::-1]
        recent_audit_rows = [
            {
                "issued_at": r["issued_at"].strftime("%Y-%m-%d %H:%M"),
                "target_timestamp": r["target_timestamp"].strftime("%Y-%m-%d %H:%M"),
                "horizon_h": int(r["horizon_h"]),
                "start_depth_mm": float(r["start_depth_mm"]),
                "pred_p50_mm": float(r["pred_p50_mm"]),
                "pred_band": f"{r['pred_p10_mm']:.0f}–{r['pred_p90_mm']:.0f} mm",
                "actual_depth_mm": float(r["actual_depth_mm"]),
                "error_mm": round(float(r["pred_p50_mm"] - r["actual_depth_mm"]), 1),
                "in_band": bool(r["pred_p10_mm"] <= r["actual_depth_mm"] <= r["pred_p90_mm"]),
                "source": str(r.get("source", "2026_backtest")),
            }
            for _, r in verified_tail.iterrows()
        ]

        selected_metrics = next(
            (m for m in horizon_scorecard if m["horizon_h"] == horizon_h),
            horizon_scorecard[0] if horizon_scorecard else {},
        )

        return {
            "selected_horizon_h": horizon_h,
            "selected_window": window,
            "data_split_metadata": data_split_metadata,
            "selected_metrics": selected_metrics,
            "horizon_scorecard": horizon_scorecard,
            "series": series,
            "recent_audit_rows": recent_audit_rows,
        }

