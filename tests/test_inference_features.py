import numpy as np
import pandas as pd

from src.data_pipeline import TIME_COLS, make_features, make_windows


def _toy_df(n=60):
    return pd.DataFrame(
        {
            "vehicle_count_scaled": np.arange(n, dtype=float),
            "hour_sin": np.zeros(n),
            "hour_cos": np.zeros(n),
            "dow_sin": np.zeros(n),
            "dow_cos": np.zeros(n),
            "is_weekend": np.zeros(n),
        }
    )


def test_make_features_matches_make_windows():
    """Inference features must be byte-for-byte what training feeds XGBoost."""
    df = _toy_df(60)
    lookback, horizon = 10, 3

    X_feat, origins = make_features(df, lookback)
    X_win, _ = make_windows(df, lookback, horizon)

    # make_windows drops the tail where the horizon has no target.
    assert X_feat.shape[0] == X_win.shape[0] + horizon
    np.testing.assert_allclose(X_feat[: X_win.shape[0]], X_win)

    # Origins are the window-end row indices.
    assert origins[0] == lookback - 1
    assert origins[-1] == len(df) - 1
    # Feature width = lags + mean/std + time encodings.
    assert X_feat.shape[1] == lookback + 2 + len(TIME_COLS)


def test_make_features_empty_when_too_short():
    df = _toy_df(5)
    X, origins = make_features(df, lookback=12)
    assert X.shape[0] == 0
    assert len(origins) == 0
