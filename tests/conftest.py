from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Enforce read-only mode before importing app modules
os.environ["OFD1_READ_ONLY"] = "1"

DATA_DIR = REPO_ROOT / "data"


def _snapshot_data_dir(data_dir: Path) -> dict[str, tuple[int, str]]:
    snap: dict[str, tuple[int, str]] = {}
    if not data_dir.exists():
        return snap
    for p in sorted(data_dir.iterdir()):
        if p.is_file():
            st = p.stat()
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            snap[p.name] = (st.st_size, digest)
    return snap


@pytest.fixture(scope="session", autouse=True)
def enforce_data_dir_immutability() -> Iterator[None]:
    """Ensure running pytest never creates, modifies, or deletes files in data/."""
    before = _snapshot_data_dir(DATA_DIR)
    yield
    after = _snapshot_data_dir(DATA_DIR)
    assert before == after, "Files in data/ were mutated during test execution!"


@pytest.fixture(scope="session")
def api_client() -> Iterator[TestClient]:
    """Session-scoped FastAPI TestClient running strictly in read-only offline mode."""
    from app.main import app, engine

    app.state.read_only = True
    engine.read_only = True
    with TestClient(app) as client:
        yield client


@pytest.fixture
def synthetic_hourly_df() -> pd.DataFrame:
    """Return 800 hours of deterministic synthetic telemetry covering dry, glitch, and flood regimes."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2026-09-01 00:00:00", periods=800, freq="1h", tz="UTC")
    idx.name = "timestamp"

    # Base quiescent depth around 530 mm with a storm pulse around hour 400..440
    depth = np.full(800, 530.0, dtype=float) + rng.normal(0.0, 1.5, size=800)
    rain = np.zeros(800, dtype=float)
    swcc_rain = np.zeros(800, dtype=float)
    stump_rain = np.zeros(800, dtype=float)

    # Corroborated storm at hours 400..408 causing a flood pulse peaking at 740 mm
    rain[400:408] = [2.5, 4.0, 6.5, 8.0, 5.5, 3.0, 1.5, 0.5]
    swcc_rain[400:408] = [2.2, 4.2, 6.8, 7.5, 5.0, 2.8, 1.2, 0.4]
    stump_rain[400:408] = [2.8, 4.5, 7.2, 8.5, 6.0, 3.2, 1.6, 0.6]

    for step in range(401, 460):
        dt = step - 400
        pulse = 220.0 * np.exp(-((dt - 8.0) ** 2) / 50.0) if dt <= 12 else 200.0 * np.exp(-(dt - 12.0) / 14.0)
        depth[step] = 530.0 + pulse

    # Inject a Stump gauge stuck-counter hardware glitch at hour 200 (stump >= 15, swcc < 5, rain < 5)
    stump_rain[200:204] = 96.6
    swcc_rain[200:204] = 0.0
    rain[200:204] = 0.0

    soil = np.clip(0.34 + 0.06 * np.sin(np.linspace(0, 4 * np.pi, 800)), 0.31, 0.42)
    soil[400:460] = 0.42

    return pd.DataFrame(
        {
            "depth_mm_raw": depth,
            "depth_mm": depth,
            "swcc_rain_mm": swcc_rain,
            "stump_rain_mm": stump_rain,
            "swcc_press_hpa": np.full(800, 992.0),
            "stump_press_hpa": np.full(800, 988.0),
            "rain_mm": rain,
            "temp_c": np.full(800, 9.5),
            "wind_kmh": np.full(800, 20.0),
            "era5_press_hpa": np.full(800, 990.0),
            "era5_soil_moisture": soil,
            "era5_soil_deep": soil,
            "et0_mm": np.full(800, 0.05),
        },
        index=idx,
    )
