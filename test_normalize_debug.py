#!/usr/bin/env python3
"""Test _normalize_score with correct pKa range."""

def _normalize_score(value, property_name, property_config):
    """Normalize a raw property value based on configuration."""
    
    config = property_config[property_name]
    
    # Extract configuration values
    value_range = config.get('range', [0, 1])
    x_min, x_max = value_range
    
    # Check for target preferred value
    if 'preferred_value' in config:
        target = config['preferred_value']
        
        # Calculate max allowed distance
        max_dist = max(target - x_min, x_max - target)
        dist_from_target = abs(value - target)
        
        print(f"    x_min={x_min}, x_max={x_max}, target={target}")
        print(f"    max_dist={max_dist}, dist_from_target={dist_from_target}")
        
        if max_dist > 0:
            # Linear score within range
            score = max(0.0, 1.0 - (dist_from_target / max_dist))
            print(f"    Linear score: {score}")
            
            # For values outside range, use softer decay
            if dist_from_target > max_dist:
                excess = dist_from_target - max_dist
                soft_decay = max(0.1, 1.0 - (excess / max_dist) * 0.9)
                score = min(score, soft_decay)
                print(f"    Outside range, soft_decay={soft_decay}")
        else:
            score = 1.0 if value == target else 0.0
        
        return score
    
    return 0.0

# Updated pKa config with correct range
property_config = {
    # LiTEN outputs pKa in roughly -2 to 3 range
    "pKa_acidic": {"range": [-2, 3], "preferred_value": 2.0},
    "pKa_basic": {"range": [-2, 3], "preferred_value": 2.0},
}

print("=" * 60)
print("Testing pKa_acidic with raw value 0.724:")
score = _normalize_score(0.724, "pKa_acidic", property_config)
print(f"Final normalized score: {score:.4f}")
print()

print("=" * 60)
print("Testing pKa_basic with raw value -0.694:")
score = _normalize_score(-0.694, "pKa_basic", property_config)
print(f"Final normalized score: {score:.4f}")
print()

print("=" * 60)
print("Testing with preferred_value = 2.0:")
score = _normalize_score(2.0, "pKa_acidic", property_config)
print(f"Normalized score for value=2.0: {score:.4f}")
