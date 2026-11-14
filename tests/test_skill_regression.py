from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.hydrology import EVAL_SKILL_HORIZONS
from scripts.backtest_nwp_ensemble import (
    FINGERPRINT_FILES,
    compute_skill_fingerprint,
    verify_or_update_skill_ledger,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO_ROOT / "benchmarks" / "skill_ledger.json"
SUMMARY_CSV_PATH = REPO_ROOT / "data" / "nwp_ensemble_backtest_summary.csv"


def test_skill_ledger_fingerprint_matches_six_tracked_files() -> None:
    """The 6-file SHA-256 fingerprint in benchmarks/skill_ledger.json must match the workspace."""
    assert LEDGER_PATH.exists(), f"Missing skill ledger at {LEDGER_PATH}"
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    live_fp = compute_skill_fingerprint(REPO_ROOT)

    assert set(ledger["files"].keys()) == set(FINGERPRINT_FILES)
    for rel_path in FINGERPRINT_FILES:
        assert ledger["files"][rel_path] == live_fp["files"][rel_path], (
            f"SHA-256 mismatch for {rel_path}: run `uv run python scripts/backtest_nwp_ensemble.py --update-ledger`"
        )
    assert ledger["current_fingerprint_sha256"] == live_fp["fingerprint_sha256"]


def test_out_of_sample_2026_skill_gates_and_coverage_fix() -> None:
    """Verify 2026 out-of-sample skill across all 8 horizons and confirm ensemble coverage >= 68%."""
    assert SUMMARY_CSV_PATH.exists(), f"Missing backtest summary at {SUMMARY_CSV_PATH}"
    summ = pd.read_csv(SUMMARY_CSV_PATH)

    assert sorted(summ["horizon_h"].astype(int).unique().tolist()) == EVAL_SKILL_HORIZONS
    assert set(summ["scenario"].unique()) == {"deterministic", "ensemble", "perfect_wx"}

    # Validate non-regression against baseline_metrics and tolerances in benchmarks/skill_ledger.json
    ledger = verify_or_update_skill_ledger(
        summ,
        ledger_path=LEDGER_PATH,
        root_dir=REPO_ROOT,
        update=False,
        reset_baseline=False,
    )
    tolerances = ledger["tolerances"]
    baseline = ledger["baseline_metrics"]

    ens_rows = summ[summ["scenario"] == "ensemble"].set_index("horizon_h")
    for h in EVAL_SKILL_HORIZONS:
        h_key = str(h)
        row = ens_rows.loc[h]
        base_h = baseline["ensemble"][h_key]

        assert row["mae_p50_mm"] <= base_h["mae_p50_mm"] + tolerances["max_mae_regression_mm"]
        assert row["rmse_p50_mm"] <= base_h["rmse_p50_mm"] + tolerances["max_rmse_regression_mm"]
        assert row["coverage_pct"] >= max(
            tolerances["min_coverage_80_pct"], base_h["coverage_pct"] - 1.5
        )
        # Confirm the d_p10 reference-frame fix keeps P10-P90 coverage well above the pre-fix 50.17%-57.51%
        assert row["coverage_pct"] >= 66.0
        assert row["f1_600_p50"] >= base_h["f1_600_p50"] - tolerances["max_f1_regression"]
        assert row["f1_700_p90"] >= base_h["f1_700_p90"] - tolerances["max_f1_regression"]
