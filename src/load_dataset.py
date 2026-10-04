"""Adapt an external traffic CSV to the pipeline's canonical format.

The training pipeline and dashboard read a single file with two columns:

    timestamp, vehicle_count

sampled at a fixed interval. This module normalizes a real-world dataset into
that schema and records the cadence + model shape in
``data/raw/dataset_meta.json`` so the rest of the pipeline (``src/config.py``)
adapts automatically.

Included preset
---------------
``metro``
    The UCI/Kaggle *Metro Interstate Traffic Volume* dataset (hourly).
    Columns: ``date_time`` -> timestamp, ``traffic_volume`` -> vehicle_count,
    resampled to 60 min with lookbacks 3 / 6 / 12 h and horizons 1 / 2 / 3 h.

What the adapter does
---------------------
1. Parses timestamps and sorts chronologically.
2. Drops duplicate timestamps. The Metro set ships thousands of rows repeated
   up to six times (verified: every duplicate shares the same ``traffic_volume``),
   so "keep first" is lossless there.
3. Resamples onto a regular grid at ``--interval-min`` using ``--agg``. Missing
   buckets stay ``NaN`` (``sum`` uses ``min_count=1``), so gaps are *visible*.
4. Keeps the largest contiguous block, splitting wherever a gap exceeds
   ``--max-gap-h``. The Metro set has a ~10-month outage in the middle and
   several multi-day gaps; forward-filling those would train the model on a
   constant, so they are cut instead. Short gaps remain ``NaN`` and are
   forward-filled by the pipeline exactly like the synthetic sensor dropouts.
5. Writes ``data/raw/traffic.csv`` and ``data/raw/dataset_meta.json``.

Examples
--------
    # Metro Interstate (CSV or CSV.GZ downloaded from Kaggle/UCI)
    python -m src.load_dataset --preset metro \\
        --input data/external/Metro_Interstate_Traffic_Volume.csv.gz

    # Any other CSV, with auto-detected columns
    python -m src.load_dataset --input my_traffic.csv \\
        --interval-min 5 --agg sum
"""

import argparse
from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, META_PATH, write_dataset_meta

PRESETS = {
    "metro": {
        "timestamp_col": "date_time",
        "value_col": "traffic_volume",
        "interval_min": 60,
        "agg": "sum",
        "lookbacks": [3, 6, 12],
        "horizon_steps": [1, 2, 3],
        "max_gap_h": 6.0,
    },
}

_TS_HINTS = ("datetime", "date_time", "timestamp", "time", "date", "ts")
_VALUE_HINTS = ("traffic_volume", "vehicle_count", "volume", "count", "flow")


def _detect_timestamp_col(df):
    for hint in _TS_HINTS:
        for col in df.columns:
            if hint in col.lower():
                return col
    # Fall back to the first column that parses cleanly as datetimes.
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() > 0.9:
            return col
    raise ValueError("Could not detect a timestamp column; pass --timestamp-col.")


def _detect_value_col(df, timestamp_col):
    for hint in _VALUE_HINTS:
        for col in df.columns:
            if col == timestamp_col:
                continue
            if hint in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
                return col
    numeric = [
        c for c in df.columns
        if c != timestamp_col and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric:
        raise ValueError("Could not detect a numeric value column; pass --value-col.")
    return numeric[0]


def _largest_contiguous_block(series, max_gap_steps):
    """Splits at NaN runs longer than `max_gap_steps`; returns the largest piece.

    The returned slice may still contain short (<= max_gap_steps) NaN gaps, which
    the training pipeline forward-fills.
    """
    isna = series.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = series.groupby(run_id).transform("size")
    big_gap = isna & (run_len > max_gap_steps)
    block_id = (big_gap & ~big_gap.shift(fill_value=False)).cumsum()

    counts = series.notna().groupby(block_id).sum()
    counts = counts[counts > 0]
    if counts.empty:
        raise ValueError("No non-missing samples after gap splitting.")
    best = counts.idxmax()
    block = series[block_id == best]
    if block.empty:
        raise ValueError("No non-missing samples after gap splitting.")
    # Trim the big-gap NaN rows that trail/lead the kept block.
    return block.loc[block.first_valid_index():block.last_valid_index()]


def load(input_path, *, preset="generic", timestamp_col=None, value_col=None,
         interval_min=None, agg=None, lookbacks=None, horizon_steps=None,
         max_gap_h=None, out_path=DATA_PATH, meta_path=META_PATH, name=None):
    """Normalizes `input_path` into the pipeline schema.

    Returns ``(frame, meta)`` where `frame` has columns
    ``["timestamp", "vehicle_count"]`` and `meta` is the manifest dict written
    to ``data/raw/dataset_meta.json``.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    preset_cfg = PRESETS.get(preset, {})
    timestamp_col = timestamp_col or preset_cfg.get("timestamp_col")
    value_col = value_col or preset_cfg.get("value_col")
    interval_min = interval_min or preset_cfg.get("interval_min", 5)
    agg = agg or preset_cfg.get("agg", "sum")
    lookbacks = lookbacks or preset_cfg.get("lookbacks", [6, 12, 24])
    horizon_steps = horizon_steps or preset_cfg.get("horizon_steps", [3, 6, 12])
    if max_gap_h is None:
        max_gap_h = preset_cfg.get("max_gap_h", 6.0)
    name = name or (preset if preset != "generic" else input_path.stem)

    raw = pd.read_csv(input_path)  # pandas transparently handles .csv.gz
    timestamp_col = timestamp_col or _detect_timestamp_col(raw)
    value_col = value_col or _detect_value_col(raw, timestamp_col)

    df = raw[[timestamp_col, value_col]].rename(
        columns={timestamp_col: "timestamp", value_col: "vehicle_count"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    n_rows = len(df)
    n_dupes = int(df["timestamp"].duplicated().sum())
    df = df.drop_duplicates(subset="timestamp", keep="first")
    series = df.set_index("timestamp")["vehicle_count"].astype(float)

    # Regular grid; keep missing buckets as NaN so gaps stay detectable.
    resampler = series.resample(f"{int(interval_min)}min")
    if agg == "sum":
        grid = resampler.sum(min_count=1)
    else:
        grid = resampler.agg(agg)

    max_gap_steps = max(1, round(float(max_gap_h) * 60 / int(interval_min)))
    block = _largest_contiguous_block(grid, max_gap_steps)

    out = block.reset_index()
    out.columns = ["timestamp", "vehicle_count"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    meta = {
        "name": str(name),
        "interval_min": int(interval_min),
        "lookback_options": [int(x) for x in lookbacks],
        "horizon_steps": [int(x) for x in horizon_steps],
        "max_gap_h": float(max_gap_h),
        "source": str(input_path),
    }
    write_dataset_meta(
        meta["name"], meta["interval_min"], meta["lookback_options"],
        meta["horizon_steps"], max_gap_h=meta["max_gap_h"], source=meta["source"],
        path=meta_path,
    )

    print(f"Input:            {input_path}")
    print(f"  rows read:      {n_rows:,}")
    print(f"  duplicate ts:   {n_dupes:,} dropped (kept first)")
    print(f"  value column:   {value_col!r}  |  timestamp column: {timestamp_col!r}")
    print(f"Resampled to:     {interval_min} min ({agg})")
    print(f"Kept largest block: {len(out):,} rows "
          f"({out['timestamp'].min()} -> {out['timestamp'].max()})")
    print(f"Wrote:            {out_path}")
    print(f"Manifest:         interval={meta['interval_min']} min, "
          f"lookbacks={meta['lookback_options']}, "
          f"horizon_steps={meta['horizon_steps']}")
    return out, meta


def _parse_int_list(text):
    if text is None:
        return None
    return [int(x) for x in str(text).replace(",", " ").split()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="Path to the source CSV/CSV.GZ")
    parser.add_argument("--preset", choices=sorted(PRESETS) + ["generic"],
                        default="generic", help="Known-column preset (default: generic)")
    parser.add_argument("--timestamp-col", default=None)
    parser.add_argument("--value-col", default=None)
    parser.add_argument("--interval-min", type=int, default=None,
                        help="Output sampling interval in minutes (default: preset/5)")
    parser.add_argument("--agg", choices=["sum", "mean", "max", "min", "first"],
                        default=None, help="Aggregation within each interval")
    parser.add_argument("--lookbacks", default=None,
                        help="Comma/space separated lookbacks in STEPS, e.g. 3,6,12")
    parser.add_argument("--horizon-steps", default=None,
                        help="Comma/space separated horizons in STEPS, e.g. 1,2,3")
    parser.add_argument("--max-gap-h", type=float, default=None,
                        help="Split at gaps longer than this many hours (default: 6)")
    parser.add_argument("--out", default=str(DATA_PATH))
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)

    load(
        args.input,
        preset=args.preset,
        timestamp_col=args.timestamp_col,
        value_col=args.value_col,
        interval_min=args.interval_min,
        agg=args.agg,
        lookbacks=_parse_int_list(args.lookbacks),
        horizon_steps=_parse_int_list(args.horizon_steps),
        max_gap_h=args.max_gap_h,
        out_path=args.out,
        name=args.name,
    )


if __name__ == "__main__":
    main()
