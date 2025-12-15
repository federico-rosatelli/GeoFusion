
import numpy as np
from src.physics import geometry

def check_constraints(R_mn, Z_mn, config_constraints):
    """
    Checks if the configuration satisfies the constraints.
    
    Constraints typically include:
    - Aspect Ratio > min_value
    - Mirror Ratio > min_value
    - Curvature limits
    
    Args:
        R_mn, Z_mn: Boundary coefficients.
        config_constraints (dict): Dictionary of constraints limits.
        
    Returns:
        dict: detailed violation info
    """
    violations = {}
    
    
    aspect_ratio = geometry.calculate_aspect_ratio(R_mn, Z_mn)
    if 'min_aspect_ratio' in config_constraints:
        if aspect_ratio < config_constraints['min_aspect_ratio']:
            violations['aspect_ratio'] = f"Value {aspect_ratio:.2f} < Min {config_constraints['min_aspect_ratio']}"
            
    
    
    return violations
