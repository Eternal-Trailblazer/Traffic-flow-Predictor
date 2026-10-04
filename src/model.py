import json
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_pipeline import process_data
from src.config import (
    HORIZONS_MIN,
    HORIZONS_STEPS,
    LOOKBACK_OPTIONS,
    MODELS_DIR,
    metrics_filename,
    model_filename,
)

def evaluate_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred))
    }

def train_and_evaluate():
    """
    Trains one XGBoost model per (lookback, horizon) combination.

    We ship models for a small set of fixed lookbacks (see LOOKBACK_OPTIONS)
    instead of retraining when the dashboard lookback slider moves.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    print(f"{'Lookback':<9} | {'Horizon':<8} | {'Model':<12} | {'MAE':<7} | {'RMSE':<7} | {'R2':<7}")
    print("-" * 70)

    for lookback in LOOKBACK_OPTIONS:
        results[f"lookback_{lookback}"] = {}

        for h_min, h_steps in zip(HORIZONS_MIN, HORIZONS_STEPS):
            # 1. Get windowed data for this specific lookback + horizon
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = process_data(
                lookback=lookback,
                horizon=h_steps
            )

            # 2. Persistence Baseline (Predict the most recent vehicle count)
            # In our feature array, the lagged counts are the first 'lookback' columns.
            last_value_idx = lookback - 1
            y_pred_persistence = X_test[:, last_value_idx]

            pers_metrics = evaluate_metrics(y_test, y_pred_persistence)
            print(f"{lookback:>5} st | {h_min:>3} mins | {'Persistence':<12} | {pers_metrics['MAE']:.3f} | {pers_metrics['RMSE']:.3f} | {pers_metrics['R2']:.3f}")

            # 3. XGBoost Model
            # n_estimators is an upper bound; early stopping on the validation
            # split picks the actual tree count. A fixed 100 trees was mildly
            # overfit -- best_iteration lands around 40-80 on this data.
            xgb = XGBRegressor(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
                eval_metric="rmse",
                early_stopping_rounds=20,
            )

            # Train on train set, validate to stop before overfitting
            xgb.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Evaluate on test set
            y_pred_xgb = xgb.predict(X_test)
            xgb_metrics = evaluate_metrics(y_test, y_pred_xgb)

            print(f"{lookback:>5} st | {h_min:>3} mins | {'XGBoost':<12} | {xgb_metrics['MAE']:.3f} | {xgb_metrics['RMSE']:.3f} | {xgb_metrics['R2']:.3f}")
            print("-" * 70)

            # Save Model (lookback/horizon-specific filename so the dashboard can
            # load by name; namespaced by interval so datasets never collide)
            model_path = MODELS_DIR / model_filename(lookback, h_min)
            joblib.dump(xgb, model_path)

            # Store results
            results[f"lookback_{lookback}"][f"horizon_{h_min}_mins"] = {
                "persistence": pers_metrics,
                "xgboost": xgb_metrics
            }

    # Save metrics to JSON
    with open(MODELS_DIR / metrics_filename(), "w") as f:
        json.dump(results, f, indent=4)
    print("Training complete. Models and metrics saved to models/ directory.")

if __name__ == "__main__":
    train_and_evaluate()