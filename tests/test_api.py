from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.engine import HORIZONS, OFD1ForecastEngine
from app.main import engine


def test_engine_init_is_side_effect_free(tmp_path: Path) -> None:
    """Instantiating OFD1ForecastEngine must not create directories or mutate disk."""
    nonexistent_dir = tmp_path / "does_not_exist_yet"
    eng = OFD1ForecastEngine(data_dir=nonexistent_dir, read_only=True)
    assert eng.read_only is True
    assert not nonexistent_dir.exists()


def test_health_endpoints(api_client: TestClient) -> None:
    """Both /health and /api/health must return 200 OK with all 9 horizons loaded in read-only mode."""
    for path in ("/health", "/api/health"):
        resp = api_client.get(path)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["sync_status"] == "ready"
        assert sorted(body["horizons_loaded"]) == HORIZONS
        assert "rain_source_active" in body
        assert "source_health" in body
        assert "nrw" in body["source_health"]
    assert engine.read_only is True


def test_dashboard_and_v1_status_forecast_history(api_client: TestClient) -> None:
    """Verify /api/dashboard, /api/v1/status, /api/v1/forecast, and /api/v1/history contracts and invariants."""
    dash_resp = api_client.get("/api/dashboard")
    assert dash_resp.status_code == 200
    dash = dash_resp.json()
    for key in (
        "generated_at_utc",
        "sync_status",
        "current",
        "nwp_ensemble",
        "gauges",
        "history_72h",
        "forecast_48h",
        "forecast_anchors",
    ):
        assert key in dash

    assert dash["current"]["alert_level"] in ("GREEN", "AMBER", "RED")
    assert len(dash["history_72h"]) == 72
    assert len(dash["forecast_48h"]) == 48
    assert len(dash["forecast_anchors"]) == len(HORIZONS) + 1

    for pt in dash["forecast_48h"]:
        assert 350.0 <= pt["p10_mm"] <= pt["p50_mm"] <= pt["p90_hydro_only_mm"] <= pt["p90_mm"]
        assert pt["status"] in ("GREEN", "AMBER", "RED")

    for anchor in dash["forecast_anchors"]:
        assert (
            350.0
            <= anchor["p10_mm"]
            <= anchor["p50_mm"]
            <= anchor["p90_hydro_only_mm"]
            <= anchor["p90_mm"]
        )

    status_resp = api_client.get("/api/v1/status")
    assert status_resp.status_code == 200
    status_body = status_resp.json()
    assert status_body["status"] == "ok"
    assert status_body["current"]["depth_mm"] == dash["current"]["depth_mm"]
    assert "depth_sensor" in status_body["gauges"]
    assert "nrw_gauge" in status_body["gauges"]

    fc_resp = api_client.get("/api/v1/forecast")
    assert fc_resp.status_code == 200
    fc_body = fc_resp.json()
    assert len(fc_body["forecast_48h"]) == 48
    assert len(fc_body["forecast_anchors"]) == len(HORIZONS) + 1
    assert "models_count" in fc_body["nwp_ensemble"]

    hist_resp = api_client.get("/api/v1/history")
    assert hist_resp.status_code == 200
    hist_body = hist_resp.json()
    assert len(hist_body["history_72h"]) == 72


def test_simulate_endpoints(api_client: TestClient) -> None:
    """Verify /api/simulate and /api/v1/simulate return 25-point trajectories and respond monotonically to storm intensity."""
    calm_resp = api_client.get(
        "/api/v1/simulate",
        params={"rain_6h_mm": 2.0, "moisture_pct": 30.0, "start_depth_mm": 520.0},
    )
    assert calm_resp.status_code == 200
    calm = calm_resp.json()
    assert len(calm["trajectory_24h"]) == 25
    assert calm["start_depth_mm"] == 520.0

    wet_resp = api_client.get(
        "/api/simulate",
        params={"rain_6h_mm": 35.0, "moisture_pct": 95.0, "start_depth_mm": 520.0},
    )
    assert wet_resp.status_code == 200
    wet = wet_resp.json()
    assert len(wet["trajectory_24h"]) == 25
    assert wet["peak_p50_mm"] > calm["peak_p50_mm"]
    assert wet["peak_p90_mm"] > calm["peak_p90_mm"]
    for step in wet["trajectory_24h"]:
        assert step["p10_mm"] <= step["p50_mm"] <= step["p90_mm"]


def test_verification_and_nwp_backtest_endpoints(api_client: TestClient) -> None:
    """Verify /api/verification, /api/v1/verification, and /api/v1/nwp-backtest return valid JSON schemas."""
    for path, h_val in (("/api/verification", 2), ("/api/v1/verification", 6)):
        v_resp = api_client.get(path, params={"horizon_h": h_val, "window": "oct_2026_floods"})
        assert v_resp.status_code == 200
        v_body = v_resp.json()
        assert v_body["selected_horizon_h"] == h_val
        assert "selected_metrics" in v_body
        assert "horizon_scorecard" in v_body
        assert len(v_body["horizon_scorecard"]) == len(HORIZONS)
        assert "series" in v_body

    bt_resp = api_client.get("/api/v1/nwp-backtest")
    assert bt_resp.status_code == 200
    bt_body = bt_resp.json()
    assert "horizons" in bt_body
    assert "records" in bt_body
    assert len(bt_body["records"]) > 0


def test_model_version_selection(monkeypatch) -> None:
    """Verify OFD1_MODEL_VERSION selects v2 by default and supports v1 rollback."""
    from app.engine import OFD1ForecastEngine

    monkeypatch.delenv("OFD1_MODEL_VERSION", raising=False)
    eng_default = OFD1ForecastEngine(data_dir="data", read_only=True)
    assert eng_default.model_version == "v2"
    assert eng_default.models_path.name == "ofd1_models_v2.joblib"

    monkeypatch.setenv("OFD1_MODEL_VERSION", "v1")
    eng_v1 = OFD1ForecastEngine(data_dir="data", read_only=True)
    assert eng_v1.model_version == "v1"
    assert eng_v1.models_path.name == "ofd1_models_v1.joblib"
