from src.congestion import get_congestion_level
from src.config import CONGESTION_THRESHOLDS

def test_get_congestion_level_boundaries():
    # We use a theoretical capacity of 100 to make the math perfectly map to percentages
    capacity = 100.0
    low_max = CONGESTION_THRESHOLDS["LOW_MAX"]
    med_max = CONGESTION_THRESHOLDS["MEDIUM_MAX"]
    
    # 1. Test Low (< low_max)
    label, color = get_congestion_level(low_max * capacity - 1e-10, capacity)
    assert label == "Low"
    assert color == "green"
    
    # 2. Test Medium boundary (Exactly low_max)
    label, color = get_congestion_level(low_max * capacity, capacity)
    assert label == "Medium"
    assert color == "amber"
    
    # 3. Test Medium range (Between low_max and med_max)
    label, color = get_congestion_level((low_max + med_max) / 2 * capacity, capacity)
    assert label == "Medium"
    assert color == "amber"
    
    # 4. Test High boundary (Exactly med_max)
    label, color = get_congestion_level(med_max * capacity, capacity)
    assert label == "High"
    assert color == "red"
    
    # 5. Test Extreme High (> med_max)
    label, color = get_congestion_level(med_max * capacity + 1, capacity)
    assert label == "High"
    assert color == "red"
    
    # 6. Test Zero Capacity Fallback
    label, color = get_congestion_level(50, 0)
    assert label == "Low"
    assert color == "green"

def test_get_congestion_level_edge_cases():
    capacity = 100.0
    low_max = CONGESTION_THRESHOLDS["LOW_MAX"]
    med_max = CONGESTION_THRESHOLDS["MEDIUM_MAX"]
    
    # Just below low boundary
    label, color = get_congestion_level(low_max * capacity - 1e-9, capacity)
    assert label == "Low"
    assert color == "green"
    
    # Just above low boundary (should be medium)
    label, color = get_congestion_level(low_max * capacity + 1e-9, capacity)
    assert label == "Medium"
    assert color == "amber"
    
    # Just below med boundary (should still be medium if >= low)
    label, color = get_congestion_level(med_max * capacity - 1e-9, capacity)
    assert label == "Medium"
    assert color == "amber"
    
    # Just above med boundary
    label, color = get_congestion_level(med_max * capacity + 1e-9, capacity)
    assert label == "High"
    assert color == "red"
    
    # Very small positive
    label, color = get_congestion_level(0.0, capacity)
    assert label == "Low"
    assert color == "green"
