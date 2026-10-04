import json

import pandas as pd

from src.config import format_duration
from src.load_dataset import load


def _write_source(path):
    """Hourly data: block A (20 rows), a 20 h gap, block B (31 raw rows)."""
    a_ts = pd.date_range("2020-01-01 00:00", periods=20, freq="1h")
    b_ts = pd.date_range("2020-01-02 16:00", periods=30, freq="1h")
    # Duplicate the first row of block B (same timestamp AND value).
    b_ts = b_ts.insert(0, b_ts[0])
    rows = (
        [{"date_time": t, "traffic_volume": 100 + i, "temp": 5.0}
         for i, t in enumerate(a_ts)]
        + [{"date_time": t, "traffic_volume": 500 + i, "temp": 7.0}
           for i, t in enumerate(b_ts)]
    )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_adapter_normalizes_schema_and_writes_meta(tmp_path):
    src = tmp_path / "raw.csv"
    out = tmp_path / "traffic.csv"
    meta = tmp_path / "dataset_meta.json"
    _write_source(src)

    frame, meta_dict = load(
        src, preset="generic", interval_min=60, agg="sum",
        lookbacks=[3, 6, 12], horizon_steps=[1, 2, 3],
        max_gap_h=6, out_path=out, meta_path=meta, name="demo",
    )

    # Canonical schema + values survive.
    assert list(frame.columns) == ["timestamp", "vehicle_count"]
    assert len(frame) == 30                      # block B, duplicate removed

    written = pd.read_csv(out, parse_dates=["timestamp"])
    assert list(written.columns) == ["timestamp", "vehicle_count"]
    assert written["timestamp"].min() == pd.Timestamp("2020-01-02 16:00")
    assert written["timestamp"].max() == pd.Timestamp("2020-01-03 21:00")
    assert written["vehicle_count"].iloc[0] == 500

    manifest = json.loads(meta.read_text())
    assert manifest["name"] == "demo"
    assert manifest["interval_min"] == 60
    assert manifest["lookback_options"] == [3, 6, 12]
    assert manifest["horizon_steps"] == [1, 2, 3]
    assert meta_dict["max_gap_h"] == 6.0


def test_adapter_drops_duplicates_and_keeps_largest_block(tmp_path):
    src = tmp_path / "raw.csv"
    out = tmp_path / "traffic.csv"
    _write_source(src)

    frame, _ = load(
        src, interval_min=60, max_gap_h=6,
        out_path=out, meta_path=tmp_path / "m.json",
    )
    # Block A (20 h) is dropped because the 20 h outage exceeds max_gap_h=6.
    assert frame["timestamp"].min() >= pd.Timestamp("2020-01-02 16:00")


def test_adapter_keeps_short_gaps_as_nan(tmp_path):
    """A gap <= max_gap_h is not split, so the block stays contiguous."""
    ts = pd.date_range("2020-01-01 00:00", periods=10, freq="1h")
    rows = [{"date_time": t, "traffic_volume": 100 + i} for i, t in enumerate(ts)]
    del rows[4]  # single missing hour -> 1-step NaN gap
    src = tmp_path / "raw.csv"
    out = tmp_path / "traffic.csv"
    pd.DataFrame(rows).to_csv(src, index=False)

    frame, _ = load(src, interval_min=60, max_gap_h=6,
                    out_path=out, meta_path=tmp_path / "m.json")
    assert len(frame) == 10
    assert int(frame["vehicle_count"].isna().sum()) == 1


def test_adapter_autodetects_columns(tmp_path):
    src = tmp_path / "raw.csv"
    _write_source(src)  # has date_time + traffic_volume, no explicit columns

    frame, _ = load(src, interval_min=60, max_gap_h=6,
                    out_path=tmp_path / "t.csv", meta_path=tmp_path / "m.json")
    assert list(frame.columns) == ["timestamp", "vehicle_count"]
    assert len(frame) > 0


def test_format_duration():
    assert format_duration(5) == "5 min"
    assert format_duration(60) == "1 h"
    assert format_duration(90) == "1 h 30 min"
    assert format_duration(180) == "3 h"
