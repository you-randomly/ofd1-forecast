#!/usr/bin/env bash
set -euo pipefail

echo "=== OFD1 Workspace Onboarding Check ==="
echo "Working directory: $(pwd)"

echo ""
echo "1. Checking required data & model artifacts in data/..."
REQUIRED_FILES=(
  "data/ofd1_aligned_hourly.parquet"
  "data/ofd1_models_v1.joblib"
  "data/forecast_verification_ledger.parquet"
  "data/phase2_trial_metrics.parquet"
  "data/horizon_breakdown_1h_to_48h.parquet"
)

for f in "${REQUIRED_FILES[@]}"; do
  if [[ -f "$f" ]]; then
    size=$(du -h "$f" | cut -f1)
    echo "  [OK] $f ($size)"
  else
    echo "  [MISSING] $f"
  fi
done

echo ""
echo "2. Checking optional local research caches in data/..."
OPTIONAL_FILES=(
  "data/ofd1_thingspeak_raw.parquet"
  "data/ofd1_aligned_15m.parquet"
  "data/historic_ofd_streamway_cleaned.csv"
  "data/open_meteo_2015_2023.parquet"
  "data/phase2_trial_predictions.parquet"
)

for f in "${OPTIONAL_FILES[@]}"; do
  if [[ -f "$f" ]]; then
    size=$(du -h "$f" | cut -f1)
    echo "  [CACHED] $f ($size)"
  else
    echo "  [NOT CACHED] $f"
  fi
done

echo ""
echo "3. Checking Python & uv virtualenv..."
uv --version
if .venv/bin/python --version >/dev/null 2>&1; then
  .venv/bin/python -c '
import pandas as pd
import joblib

df = pd.read_parquet("data/ofd1_aligned_hourly.parquet")
models = joblib.load("data/ofd1_models_v1.joblib")
print(f"  Aligned hourly rows  : {len(df):,} ({df.index.min()} -> {df.index.max()})")
print(f"  Valid depth hours    : {df[\"depth_mm\"].notna().sum():,}")
print(f"  Loaded model horizons: {sorted(models.keys())}")
'
else
  echo "  [NOTE] .venv/bin/python points to ~/.local/share/uv/python/ (outside default sandbox)."
  echo "         Run 'uv run ...' commands with BypassSandbox: true when executing Python."
fi

echo ""
echo "=== Onboarding check complete ==="
