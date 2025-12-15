
import numpy as np

def get_surface_coordinates(config_data, theta_res=64, phi_res=128):
    """
    Calculates the Cartesian coordinates (X, Y, Z) and partial derivatives
    of the plasma boundary surface from Fourier coefficients.
    
    Args:
        config_data (dict): Dictionary containing boundary coefficients.
        theta_res (int): Resolution in poloidal direction.
        phi_res (int): Resolution in toroidal direction.
        
    Returns:
        dict: containing 'X', 'Y', 'Z', 'dX_dtheta', 'dX_dphi', etc.
    """
    R_mn_matrix = np.array(config_data['boundary.r_cos'])
    Z_mn_matrix = np.array(config_data['boundary.z_sin'])
    
    N_fp = config_data['boundary.n_field_periods']

    poloidal_modes_m = np.arange(R_mn_matrix.shape[0])
    
    n_columns = R_mn_matrix.shape[1]
    toroidal_modes_n = np.arange(-(n_columns // 2), (n_columns // 2) + 1)

    theta = np.linspace(0, 2 * np.pi, theta_res, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, phi_res, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij') 

    R = np.zeros_like(theta_grid)
    Z = np.zeros_like(theta_grid)
    
    
    dR_dtheta = np.zeros_like(theta_grid)
    dR_dphi = np.zeros_like(theta_grid)
    dZ_dtheta = np.zeros_like(theta_grid)
    dZ_dphi = np.zeros_like(theta_grid)

    for m_index, m in enumerate(poloidal_modes_m):
        for n_index, n in enumerate(toroidal_modes_n):
            R_c = R_mn_matrix[m_index, n_index]
            Z_c = Z_mn_matrix[m_index, n_index]
            
            angle = m * theta_grid - n * N_fp * phi_grid
            
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            R += R_c * cos_angle
            Z += Z_c * sin_angle
            
            
            
            
            
            dR_dtheta -= R_c * sin_angle * m
            
            dR_dphi += R_c * sin_angle * (n * N_fp)
            
            
            dZ_dtheta += Z_c * cos_angle * m
            
            dZ_dphi -= Z_c * cos_angle * (n * N_fp)

    
    
    
    
    
    X = R * np.cos(phi_grid)
    Y = R * np.sin(phi_grid)
    
    
    
    dX_dtheta = dR_dtheta * np.cos(phi_grid)
    
    dX_dphi = dR_dphi * np.cos(phi_grid) - R * np.sin(phi_grid)
    
    
    dY_dtheta = dR_dtheta * np.sin(phi_grid)
    
    dY_dphi = dR_dphi * np.sin(phi_grid) + R * np.cos(phi_grid)
    
    return {
        'X': X, 'Y': Y, 'Z': Z,
        'dX_dtheta': dX_dtheta, 'dX_dphi': dX_dphi,
        'dY_dtheta': dY_dtheta, 'dY_dphi': dY_dphi,
        'dZ_dtheta': dZ_dtheta, 'dZ_dphi': dZ_dphi,
        'R': R
    }


def calculate_aspect_ratio(R_mn, Z_mn):
    """
    Calculates the aspect ratio.
    """
    n_columns = R_mn.shape[1]
    n_0_index = n_columns // 2
    
    major_radius = R_mn[0, n_0_index]
    
    if R_mn.shape[0] > 1:
        minor_radius = (R_mn[1, n_0_index] + Z_mn[1, n_0_index]) / 2.0
    else:
        minor_radius = 1.0 
        
    return abs(major_radius / minor_radius)

def calculate_mirror_ratio(B_field):
    if B_field is None or B_field.size == 0:
        return 0.0
        
    b_max = np.max(B_field)
    b_min = np.min(B_field)
    
    return (b_max - b_min) / (b_max + b_min)

def calculate_curvature(surface_data):
    """
    Calculates Mean (H) and Gaussian (K) curvature.
    
    Args:
        surface_data (dict): Output from get_surface_coordinates.
        
    Returns:
        dict: {'H': Mean Curvature, 'K': Gaussian Curvature, 'Normal': normal vectors}
    """
    Xu = surface_data['dX_dtheta']
    Xv = surface_data['dX_dphi']
    Yu = surface_data['dY_dtheta']
    Yv = surface_data['dY_dphi']
    Zu = surface_data['dZ_dtheta']
    Zv = surface_data['dZ_dphi']

    
    E = Xu**2 + Yu**2 + Zu**2
    F = Xu*Xv + Yu*Yv + Zu*Zv
    G = Xv**2 + Yv**2 + Zv**2
    
    
    
    Nx = Yu*Zv - Zu*Yv
    Ny = Zu*Xv - Xu*Zv
    Nz = Xu*Yv - Yu*Xv
    
    Norm = np.sqrt(Nx**2 + Ny**2 + Nz**2)
    
    nx = Nx / Norm
    ny = Ny / Norm
    nz = Nz / Norm
    
    
    Xuu = np.gradient(Xu, axis=0) 
    Xuv = np.gradient(Xu, axis=1) 
    Xvv = np.gradient(Xv, axis=1) 
    
    Yuu = np.gradient(Yu, axis=0)
    Yuv = np.gradient(Yu, axis=1)
    Yvv = np.gradient(Yv, axis=1)
    
    Zuu = np.gradient(Zu, axis=0)
    Zuv = np.gradient(Zu, axis=1)
    Zvv = np.gradient(Zv, axis=1)
    
    
    
    L = nx*Xuu + ny*Yuu + nz*Zuu
    
    M = nx*Xuv + ny*Yuv + nz*Zuv
    
    N_coeff = nx*Xvv + ny*Yvv + nz*Zvv
    
    
    
    
    
    denom = E*G - F**2
    K = (L*N_coeff - M**2) / denom
    H = (E*N_coeff + G*L - 2*F*M) / (2*denom)
    
    return {'H': H, 'K': K, 'Normal': np.stack([nx, ny, nz], axis=-1)}
