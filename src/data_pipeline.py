import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def add_time_features(df):
    """Generates cyclic encodings for hour and day of week."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    hour = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
    dow = df['timestamp'].dt.dayofweek
    
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    df['is_weekend'] = (dow >= 5).astype(int)
    
    return df

# Time features are taken at the END of the lookback window.
TIME_COLS = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']


def build_feature_row(window_counts, current_time_feats):
    """Builds one XGBoost feature row: lags + rolling stats + time encodings."""
    window_counts = np.asarray(window_counts, dtype=float)
    return np.concatenate([
        window_counts,
        [np.mean(window_counts), np.std(window_counts)],
        np.asarray(current_time_feats, dtype=float),
    ])


def make_features(df, lookback, target_col='vehicle_count_scaled'):
    """
    Builds the feature matrix X for every origin index where a full lookback
    window exists, WITHOUT requiring a future target.

    Returns:
        X: array of shape (n_samples, lookback + 2 + 5)
        origins: array of the window-end row indices, so that row i of X
                 corresponds to df.iloc[origins[i]] as the "now" position.
    """
    target_series = df[target_col].values
    time_features = df[TIME_COLS].values

    X, origins = [], []
    for i in range(len(df) - lookback + 1):
        end = i + lookback - 1
        X.append(build_feature_row(target_series[i:end + 1], time_features[end]))
        origins.append(end)

    if not X:
        return np.empty((0, lookback + 2 + len(TIME_COLS))), np.array([], dtype=int)

    return np.array(X), np.array(origins)


def make_windows(df, lookback, horizon, target_col='vehicle_count_scaled'):
    """
    Creates tabular features for XGBoost.
    X contains the lagged counts, rolling stats, and time features at the window's end.
    y is the target value `horizon` steps ahead.
    """
    target_series = df[target_col].values
    time_features = df[TIME_COLS].values

    # Ensure we don't iterate past the available data for the target horizon
    max_idx = len(df) - lookback - horizon + 1

    X, y = [], []
    for i in range(max_idx):
        end = i + lookback - 1
        # Flatten into a single row for Tabular ML (XGBoost)
        X.append(build_feature_row(target_series[i:end + 1], time_features[end]))
        # Target value `horizon` steps after the window ends
        y.append(target_series[end + horizon])

    return np.array(X), np.array(y)

def process_data(lookback=12, horizon=3):
    """Main pipeline execution for training."""
    df = pd.read_csv("data/raw/traffic.csv")
    df = add_time_features(df)
    
    # Handle the missing values injected in Phase 1 using forward fill
    df['vehicle_count'] = df['vehicle_count'].ffill()
    
    # Chronological Split: 70% Train, 15% Val, 15% Test
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Scale only on train to prevent leakage
    scaler = StandardScaler()
    train_df['vehicle_count_scaled'] = scaler.fit_transform(train_df[['vehicle_count']])
    val_df['vehicle_count_scaled'] = scaler.transform(val_df[['vehicle_count']])
    test_df['vehicle_count_scaled'] = scaler.transform(test_df[['vehicle_count']])
    
    # Save the scaler for inference in the Streamlit app
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    
    # Generate sliding windows separately to prevent boundary leakage
    X_train, y_train = make_windows(train_df, lookback, horizon)
    X_val, y_val = make_windows(val_df, lookback, horizon)
    X_test, y_test = make_windows(test_df, lookback, horizon)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)