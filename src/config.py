# src/config.py

# Time interval configurations
SAMPLING_INTERVAL_MIN = 5
LOOKBACK_STEPS_DEFAULT = 12  # 1 hour (12 * 5 mins)

# Fixed lookbacks (in 5-min steps) that we train and ship models for.
# The dashboard lookback slider picks among these; we do not retrain on the fly.
LOOKBACK_OPTIONS = [6, 12, 24]  # 30 mins, 1 hour, 2 hours

# Prediction horizons (Minutes and corresponding steps)
HORIZONS_MIN = [15, 30, 60]
HORIZONS_STEPS = [3, 6, 12]

# Congestion thresholds (Values to be tuned later based on capacity percentiles)
CONGESTION_THRESHOLDS = {
    "LOW_MAX": 0.40,     # Below 40% of 95th percentile capacity
    "MEDIUM_MAX": 0.70   # 40% to 70% of 95th percentile capacity
}