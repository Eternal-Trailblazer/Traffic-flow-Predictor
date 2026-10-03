import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

def generate(days=90, seed=42):
    """Generates synthetic 5-minute interval traffic data."""
    np.random.seed(seed)
    
    # Create time index
    start_date = pd.Timestamp("2026-01-01 00:00:00")
    periods = days * 24 * 12  # 12 intervals per hour
    timestamps = pd.date_range(start_date, periods=periods, freq='5min')
    
    df = pd.DataFrame({'timestamp': timestamps})
    df['hour'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    
    # Base traffic volume functions
    def get_weekday_volume(hour):
        # Double peak: 08:30 and 17:30
        morning_peak = 300 * np.exp(-0.5 * ((hour - 8.5) / 1.5)**2)
        evening_peak = 350 * np.exp(-0.5 * ((hour - 17.5) / 2.0)**2)
        baseline = 50 + 20 * np.sin(np.pi * hour / 12) # Slight daytime rise
        return baseline + morning_peak + evening_peak
        
    def get_weekend_volume(hour):
        # Single flatter peak around 13:00
        midday_peak = 200 * np.exp(-0.5 * ((hour - 13.0) / 4.0)**2)
        baseline = 30 + 10 * np.sin(np.pi * hour / 12)
        return baseline + midday_peak
    
    # Apply base volumes
    df['base_count'] = np.where(
        df['is_weekend'],
        df['hour'].apply(get_weekend_volume),
        df['hour'].apply(get_weekday_volume)
    )
    
    # Add Poisson noise and cast to float to prevent Pandas type errors
    df['vehicle_count'] = np.random.poisson(df['base_count']).astype(float)
    
    # Inject Anomalies (Accidents / Events)
    n_anomalies = int(days * 0.5) # About 1 anomaly every 2 days
    anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['accident', 'event'])
        duration = np.random.randint(6, 18) # 30 to 90 mins
        
        if anomaly_type == 'accident':
            # Sudden drop in traffic
            df.loc[idx:idx+duration, 'vehicle_count'] = df.loc[idx:idx+duration, 'vehicle_count'] * 0.2
        else:
            # Sudden spike (event letting out)
            df.loc[idx:idx+duration, 'vehicle_count'] = df.loc[idx:idx+duration, 'vehicle_count'] * 2.5
            
    # Inject a few missing values (simulating sensor downtime)
    n_missing = int(len(df) * 0.005) # 0.5% missing data
    missing_indices = np.random.choice(df.index, n_missing, replace=False)
    df.loc[missing_indices, 'vehicle_count'] = np.nan
    
    # Clean up and ensure integer counts
    df['vehicle_count'] = df['vehicle_count'].round()
    
    return df[['timestamp', 'vehicle_count']]

def stream(df):
    """Generator yielding one row at a time for live Streamlit simulation."""
    for _, row in df.iterrows():
        yield row

if __name__ == "__main__":
    print("Generating 90 days of traffic data...")
    traffic_df = generate(days=90)
    
    # Save to raw data folder
    traffic_df.to_csv("data/raw/traffic.csv", index=False)
    print("Saved to data/raw/traffic.csv")
    
    # Visual check for the first 7 days (as requested by the plan's 'Done when' condition)
    print("Plotting first week for visual verification...")
    week_df = traffic_df.head(7 * 24 * 12)
    
    plt.figure(figsize=(15, 5))
    plt.plot(week_df['timestamp'], week_df['vehicle_count'], label='Vehicle Count', color='blue', alpha=0.7)
    plt.title("Traffic Flow - 1 Week Sample (Notice the Weekday Double Peaks vs Weekend Curve)")
    plt.xlabel("Date")
    plt.ylabel("Vehicles per 5-min interval")
    plt.grid(True)
    plt.tight_layout()
    plt.show()