import numpy as np
import pandas as pd
from src.data_pipeline import make_windows

def test_make_windows_shapes_and_logic():
    # Toy dataframe with 100 sequential numbers (arange)
    df = pd.DataFrame({
        'vehicle_count_scaled': np.arange(100.0),
        'hour_sin': np.zeros(100),
        'hour_cos': np.zeros(100),
        'dow_sin': np.zeros(100),
        'dow_cos': np.zeros(100),
        'is_weekend': np.zeros(100)
    })
    
    lookback = 10
    horizon = 3
    
    X, y = make_windows(df, lookback=lookback, horizon=horizon, target_col='vehicle_count_scaled')
    
    # 1. Output shapes test
    expected_samples = 100 - lookback - horizon + 1 
    assert X.shape[0] == expected_samples
    assert y.shape[0] == expected_samples
    
    # Features = 10 lags + 2 rolling stats + 5 time encodings = 17 features total
    assert X.shape[1] == lookback + 2 + 5
    
    # 2. Logic test on a toy arange series
    # First window features should be index 0 through 9
    np.testing.assert_array_equal(X[0][:lookback], np.arange(10.0))
    # Window 0 ends at index 9. Horizon 3 means the target is index 9 + 3 = 12.
    assert y[0] == 12.0

def test_no_leakage_between_splits():
    # Verify no test index appears in train window logic
    train_df = pd.DataFrame({'vehicle_count_scaled': np.arange(0.0, 50.0),
                             'hour_sin': np.zeros(50), 'hour_cos': np.zeros(50),
                             'dow_sin': np.zeros(50), 'dow_cos': np.zeros(50), 'is_weekend': np.zeros(50)})
                             
    test_df = pd.DataFrame({'vehicle_count_scaled': np.arange(50.0, 100.0),
                            'hour_sin': np.zeros(50), 'hour_cos': np.zeros(50),
                            'dow_sin': np.zeros(50), 'dow_cos': np.zeros(50), 'is_weekend': np.zeros(50)})
    
    X_train, y_train = make_windows(train_df, lookback=5, horizon=1)
    X_test, y_test = make_windows(test_df, lookback=5, horizon=1)
    
    # Max count in train is 49. It should never see anything 50 or above.
    assert np.max(X_train[:, :5]) <= 49.0
    assert np.max(y_train) <= 49.0
    
    # Min count in test is 50. It should never see anything 49 or below.
    assert np.min(X_test[:, :5]) >= 50.0
    assert np.min(y_test) >= 50.0