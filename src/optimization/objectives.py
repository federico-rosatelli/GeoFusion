
import numpy as np
from src.physics import geometry, magnetic

def calculate_coil_simplicity(R_mn, Z_mn, config_data_template):
    """
    Objectives: Simple-to-build QI
    Metric: e_L_nablaB (Normalized magnetic field gradient scale length)
    
    Args:
        R_mn, Z_mn: Boundary coefficients.
        config_data_template (dict): Template with metadata (N_fp, etc).
        
    Returns:
        float: The objective value (lower is simpler coils).
    """
    
    config = config_data_template.copy()
    config['boundary.r_cos'] = R_mn
    config['boundary.z_sin'] = Z_mn
    
    
    surface_data = geometry.get_surface_coordinates(config)
    
    
    B_field = magnetic.calculate_magnetic_field_proxy(surface_data)
    grad_B_norm = magnetic.calculate_magnetic_field_gradient(B_field, surface_data)
    
    
    B_safe = np.where(B_field == 0, 1e-12, B_field)
    inverse_scale_length = grad_B_norm / np.abs(B_safe)
    
    
    return np.mean(inverse_scale_length)


def calculate_mhd_stability(R_mn, Z_mn, config_data_template):
    """
    Objectives: W_MHD (Vacuum Magnetic Well)
    
    Args:
        R_mn, Z_mn:  Boundary coefficients.
    Returns:
        float: Stability metric.
    """
    
    
    
    
    config = config_data_template.copy()
    config['boundary.r_cos'] = R_mn
    config['boundary.z_sin'] = Z_mn
    
    surface_data = geometry.get_surface_coordinates(config)
    curvature_data = geometry.calculate_curvature(surface_data)
    
    
    
    
    return np.mean(curvature_data['H']**2)


# TODO: add Vacuum Magnetic Well (W_MHD)