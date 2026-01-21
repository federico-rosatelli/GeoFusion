
import numpy as np

def calculate_magnetic_field_proxy(surface_data):   # TODO: Replace with actual magnetic field
    """
    Calculates a proxy magnetic field on the surface.
    
    Assumption:
    Toroidal field dominates B ~ B0 * R0 / R
    
    Args:
        surface_data (dict): Surface coordinates (X, Y, Z, R, etc).
        
    Returns:
        np.ndarray: B field magnitude.
    """
    R = surface_data['R']
    
    R_safe = np.where(R == 0, 1.0, R)
    B_mod = 1.0 / R_safe
    
    return B_mod

def calculate_magnetic_field_gradient(B_field, surface_data):
    """
    Calculates the gradient of the magnetic field magnitude on the surface.
    
    e_L_nablaB = || grad(|B|) ||
    
    Args:
        B_field (np.ndarray): Magnetic field magnitude on the surface grid.
        surface_data (dict): Surface geometric data.
        
    Returns:
        np.ndarray: Gradient magnitude field.
    """
    
    
    
    Bu = np.gradient(B_field, axis=0)
    Bv = np.gradient(B_field, axis=1)
    
    
    Xu = surface_data['dX_dtheta']
    Xv = surface_data['dX_dphi']
    Yu = surface_data['dY_dtheta']
    Yv = surface_data['dY_dphi']
    Zu = surface_data['dZ_dtheta']
    Zv = surface_data['dZ_dphi']

    E = Xu**2 + Yu**2 + Zu**2
    F = Xu*Xv + Yu*Yv + Zu*Zv
    G = Xv**2 + Yv**2 + Zv**2
    
    denom = E*G - F**2
    
    denom = np.where(denom < 1e-12, 1e-12, denom)
    
    grad_norm_sq = (G * Bu**2 - 2*F * Bu * Bv + E * Bv**2) / denom
    grad_norm = np.sqrt(np.abs(grad_norm_sq))
    
    return grad_norm
