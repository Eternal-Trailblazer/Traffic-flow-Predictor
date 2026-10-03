from src.config import CONGESTION_THRESHOLDS

def get_congestion_level(predicted_count, capacity_95th):
    """
    Maps a predicted vehicle count to a congestion level and color 
    based on the road's 95th percentile observed capacity.
    """
    if capacity_95th <= 0:
        return "Low", "green"  # Safe fallback to prevent division by zero
        
    ratio = predicted_count / capacity_95th
    
    # Below 40% of capacity
    if ratio < CONGESTION_THRESHOLDS["LOW_MAX"]:
        return "Low", "green"
    # Between 40% and 70% of capacity
    elif ratio < CONGESTION_THRESHOLDS["MEDIUM_MAX"]:
        return "Medium", "amber"
    # 70% and above
    else:
        return "High", "red"
    