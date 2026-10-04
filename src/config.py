# src/config.py
"""Central configuration.

The pipeline can run against datasets sampled at different cadences. If
``data/raw/dataset_meta.json`` exists (written by ``src.mock_stream`` or
``src.load_dataset``) it selects the dataset name, sampling interval, lookbacks
and horizons; otherwise the synthetic 5-minute defaults below apply.

Artifact filenames embed the sampling interval, so a 5-minute dataset and an
hourly one can be trained side by side without overwriting each other.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Canonical paths used by the pipeline and the dashboard.
DATA_PATH = ROOT / "data" / "raw" / "traffic.csv"
META_PATH = ROOT / "data" / "raw" / "dataset_meta.json"
MODELS_DIR = ROOT / "models"

# Defaults describe the synthetic 5-minute dataset.
SYNTHETIC_NAME = "synthetic"
SYNTHETIC_INTERVAL_MIN = 5
SYNTHETIC_LOOKBACKS = [6, 12, 24]      # 30 min, 1 h, 2 h
SYNTHETIC_HORIZON_STEPS = [3, 6, 12]   # 15, 30, 60 min


def _load_meta():
    """Reads the optional dataset manifest, falling back to synthetic defaults."""
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            interval = int(meta["interval_min"])
            lookbacks = sorted({int(x) for x in meta["lookback_options"]})
            steps = [int(x) for x in meta["horizon_steps"]]
            name = str(meta.get("name", "custom"))
            if interval > 0 and lookbacks and steps and all(s > 0 for s in steps):
                return name, interval, lookbacks, steps
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            # A malformed manifest should never break the app/tests.
            pass
    return (
        SYNTHETIC_NAME,
        SYNTHETIC_INTERVAL_MIN,
        list(SYNTHETIC_LOOKBACKS),
        list(SYNTHETIC_HORIZON_STEPS),
    )


DATASET_NAME, SAMPLING_INTERVAL_MIN, LOOKBACK_OPTIONS, HORIZONS_STEPS = _load_meta()

# Human-readable horizons, derived so they always match the cadence.
HORIZONS_MIN = [steps * SAMPLING_INTERVAL_MIN for steps in HORIZONS_STEPS]

LOOKBACK_STEPS_DEFAULT = LOOKBACK_OPTIONS[len(LOOKBACK_OPTIONS) // 2]

# How many samples make up an hour (used for chart-history windows).
STEPS_PER_HOUR = max(1, round(60 / SAMPLING_INTERVAL_MIN))

# Congestion thresholds (ratios of the 95th-percentile capacity).
CONGESTION_THRESHOLDS = {
    "LOW_MAX": 0.40,     # Below 40% of 95th percentile capacity
    "MEDIUM_MAX": 0.70   # 40% to 70% of 95th percentile capacity
}


def write_dataset_meta(name, interval_min, lookback_options, horizon_steps,
                       max_gap_h=None, source=None, path=META_PATH):
    """Writes the manifest that makes the rest of the pipeline cadence-aware."""
    meta = {
        "name": str(name),
        "interval_min": int(interval_min),
        "lookback_options": [int(x) for x in lookback_options],
        "horizon_steps": [int(x) for x in horizon_steps],
    }
    if max_gap_h is not None:
        meta["max_gap_h"] = float(max_gap_h)
    if source is not None:
        meta["source"] = str(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def model_filename(lookback, horizon_min):
    """Lookback/horizon-specific model file, namespaced by sampling interval."""
    return f"xgb_lb{lookback}_h{horizon_min}_i{SAMPLING_INTERVAL_MIN}.joblib"


def scaler_filename():
    return f"scaler_i{SAMPLING_INTERVAL_MIN}.joblib"


def metrics_filename():
    return f"metrics_i{SAMPLING_INTERVAL_MIN}.json"


def format_duration(minutes):
    """Human label for a duration in minutes: 15 -> '15 min', 60 -> '1 h'."""
    minutes = int(minutes)
    hours, mins = divmod(minutes, 60)
    if hours and mins:
        return f"{hours} h {mins} min"
    if hours:
        return f"{hours} h"
    return f"{mins} min"
