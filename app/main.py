from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.engine import OFD1ForecastEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ofd1.web")

DATA_DIR = Path(os.environ.get("OFD1_DATA_DIR", "data"))
REFRESH_INTERVAL_SECONDS = int(os.environ.get("OFD1_REFRESH_SECONDS", "600"))
STATIC_DIR = Path(__file__).parent / "static"

engine = OFD1ForecastEngine(data_dir=DATA_DIR)
_diagnostics_cache: dict[str, Any] | None = None


async def _periodic_sync_loop() -> None:
    """Background loop that syncs ThingSpeak + Open-Meteo every 10 minutes."""
    while True:
        try:
            await engine.refresh_live_telemetry()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Periodic telemetry sync error: %s", exc)
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _diagnostics_cache
    read_only = bool(getattr(app.state, "read_only", False)) or (
        os.environ.get("OFD1_READ_ONLY", "0") == "1"
    )
    engine.initialize(read_only=read_only)
    _diagnostics_cache = engine.get_diagnostics_summary()
    sync_task: asyncio.Task[None] | None = None
    if not read_only:
        sync_task = asyncio.create_task(_periodic_sync_loop())
    yield
    if sync_task is not None:
        sync_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sync_task


app = FastAPI(
    title="OFD1 Streamway Caver Forecast & Telemetry API",
    description="Real-time karst hydrology nowcasting & 48-hour flood pulse warning system for Ogof Ffynnon Ddu 1.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "sync_status": engine.sync_status,
        "last_sync_utc": engine.last_sync_utc,
        "horizons_loaded": list(engine.models.keys()),
    }


@app.get("/api/dashboard")
async def get_dashboard() -> dict[str, Any]:
    return engine.latest_payload


@app.get("/api/v1/status")
async def get_v1_status() -> dict[str, Any]:
    return {
        "status": "ok",
        "sync_status": engine.sync_status,
        "generated_at_utc": engine.latest_payload.get("generated_at_utc"),
        "current": engine.latest_payload.get("current", {}),
        "gauges": engine.latest_payload.get("gauges", {}),
    }


@app.get("/api/v1/forecast")
async def get_v1_forecast() -> dict[str, Any]:
    return {
        "generated_at_utc": engine.latest_payload.get("generated_at_utc"),
        "current": engine.latest_payload.get("current", {}),
        "nwp_ensemble": engine.latest_payload.get("nwp_ensemble", {}),
        "forecast_48h": engine.latest_payload.get("forecast_48h", []),
        "forecast_anchors": engine.latest_payload.get("forecast_anchors", []),
    }


@app.get("/api/v1/history")
async def get_v1_history() -> dict[str, Any]:
    return {
        "generated_at_utc": engine.latest_payload.get("generated_at_utc"),
        "history_72h": engine.latest_payload.get("history_72h", []),
    }


@app.post("/api/refresh")
async def force_refresh() -> dict[str, Any]:
    return await engine.refresh_live_telemetry()


@app.get("/api/simulate")
@app.get("/api/v1/simulate")
async def simulate_storm(
    rain_6h_mm: float = Query(15.0, ge=0.0, le=60.0, description="6-hour cumulative rainfall (mm)"),
    moisture_pct: float = Query(80.0, ge=5.0, le=120.0, description="Peat bog saturation index (%)"),
    start_depth_mm: float | None = Query(None, ge=380.0, le=2000.0, description="Initial streamway depth (mm)"),
) -> dict[str, Any]:
    return engine.simulate_what_if(
        rain_6h_mm=rain_6h_mm,
        moisture_pct=moisture_pct,
        start_depth_mm=start_depth_mm,
    )


@app.get("/api/diagnostics")
async def get_diagnostics() -> dict[str, Any]:
    global _diagnostics_cache
    if _diagnostics_cache is None:
        _diagnostics_cache = engine.get_diagnostics_summary()
    return _diagnostics_cache


@app.get("/api/verification")
@app.get("/api/v1/verification")
async def get_verification(
    horizon_h: int = Query(2, description="Forecast lead horizon in hours (1, 2, 3, 6, 12, 18, 24, 36, 48)"),
    window: str = Query("oct_2026_floods", description="Evaluation time window"),
) -> dict[str, Any]:
    if horizon_h not in [1, 2, 3, 6, 12, 18, 24, 36, 48]:
        horizon_h = 2
    return engine.get_verification_history(horizon_h=horizon_h, window=window)


@app.get("/api/v1/nwp-backtest")
async def get_v1_nwp_backtest() -> dict[str, Any]:
    if not engine.ensemble_backtest_summary_path.exists():
        return {"horizons": [], "records": []}
    df = pd.read_csv(engine.ensemble_backtest_summary_path)
    horizons = sorted(df["horizon_h"].astype(int).unique().tolist()) if "horizon_h" in df.columns else []
    records = df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
    return {
        "horizons": horizons,
        "records": records,
    }


@app.get("/api/forecast-evolution")
async def get_forecast_evolution(
    target_timestamp: str = Query(..., description="Target arrival timestamp in ISO format (e.g. 2026-11-09T16:00:00Z)"),
) -> dict[str, Any]:
    return engine.get_forecast_evolution_for_target(target_iso=target_timestamp)

