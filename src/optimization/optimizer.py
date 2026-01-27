from src.physics import geometry
from src.optimization import objectives
import numpy as np
from scipy.optimize import minimize
from src.ml.surrogate import StellaratorSurrogate
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def optimize_stellarator(initial_config, problem_type="simple-to-build", max_iter=100):
    """
    Main optimization loop using scipy.optimize.
    
    Args:
        initial_config (dict): Initial boundary coefficients and metadata.
        problem_type (str): "simple-to-build" or "mhd-stable".
        max_iter (int): Maximum number of iterations (generations or steps).
        
    Returns:
        dict: Optimized configuration.
    """
    print(f"Starting optimization for {problem_type}...")
    
    
    R_mn_init = np.array(initial_config['boundary.r_cos'])
    Z_mn_init = np.array(initial_config['boundary.z_sin'])
    
    shape_R = R_mn_init.shape
    shape_Z = Z_mn_init.shape
    
    target_volume = geometry.calculate_volume(R_mn_init, Z_mn_init)
    x0 = np.concatenate([R_mn_init.flatten(), Z_mn_init.flatten()])

    
    model = StellaratorSurrogate(input_shape=x0.shape, hidden_dims=[1024, 512, 256, 128])
    model.load_state_dict(torch.load("models/surrogate/stellarator_surrogate.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    def reshape_coeffs(x):
        split = R_mn_init.size
        R_flat = x[:split]
        Z_flat = x[split:]
        return R_flat.reshape(shape_R), Z_flat.reshape(shape_Z)
    
    
    def loss_function(x):
        R_mn, Z_mn = reshape_coeffs(x)
        
        
        if problem_type == "GeoFusion-nn":
            objectives.calculate_geo_fusion_nn(R_mn, Z_mn, model)

        elif problem_type == "simple-to-build":
            
            val = objectives.calculate_coil_simplicity(R_mn, Z_mn, initial_config)
        else:
            
            val = objectives.calculate_mhd_stability(R_mn, Z_mn, initial_config)
            
        
        
        reg = 0.01 * (np.sum(x**2))
        
        return val + reg


    
    bounds = [(-5.0, 5.0) for _ in range(len(x0))]
    
    def min_aspect_ratio_constraint(x):
        R_mn, Z_mn = reshape_coeffs(x)
        ar = geometry.calculate_aspect_ratio(R_mn, Z_mn)
        return ar - 6.0 

    
    def volume_constraint_lower(x):
        R_mn, Z_mn = reshape_coeffs(x)
        vol = geometry.calculate_volume(R_mn, Z_mn)
        return vol - (target_volume * 0.95)

    def volume_constraint_upper(x):
        R_mn, Z_mn = reshape_coeffs(x)
        vol = geometry.calculate_volume(R_mn, Z_mn)
        return (target_volume * 1.05) - vol
    
    def max_mirror_ratio_constraint(x):
        R_mn, Z_mn = reshape_coeffs(x)
        mr = geometry.calculate_geometric_mirror_ratio(R_mn, Z_mn)
        TARGET_MR = 0.08
        return TARGET_MR - mr
        
    cons = (
        {'type': 'ineq', 'fun': min_aspect_ratio_constraint},
        {'type': 'ineq', 'fun': volume_constraint_lower},
        {'type': 'ineq', 'fun': volume_constraint_upper},
        {'type': 'ineq', 'fun': max_mirror_ratio_constraint}
    )
    
    
    history = []
    def callback(x):
        val = loss_function(x)
        history.append(val)
    
    
    
    res = minimize(
        loss_function, 
        x0, 
        method='SLSQP', 
        bounds=bounds,
        constraints=cons,
        callback=callback,
        options={'maxiter': max_iter, 'disp': True}
    )
    
    
    
    R_opt, Z_opt = reshape_coeffs(res.x)
    optimized_config = initial_config.copy()
    optimized_config['boundary.r_cos'] = R_opt.tolist()
    optimized_config['boundary.z_sin'] = Z_opt.tolist()
    
    
    optimized_config['optimization_history'] = history
    
    return optimized_config
