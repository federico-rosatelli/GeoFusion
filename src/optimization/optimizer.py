
from src.optimization import objectives, constraints
import numpy as np
from scipy.optimize import minimize

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
    
    
    x0 = np.concatenate([R_mn_init.flatten(), Z_mn_init.flatten()])
    
    def reshape_coeffs(x):
        split = R_mn_init.size
        R_flat = x[:split]
        Z_flat = x[split:]
        return R_flat.reshape(shape_R), Z_flat.reshape(shape_Z)
    
    
    def loss_function(x):
        R_mn, Z_mn = reshape_coeffs(x)
        
        
        if problem_type == "simple-to-build":
            
            val = objectives.calculate_coil_simplicity(R_mn, Z_mn, initial_config)
        else:
            
            val = objectives.calculate_mhd_stability(R_mn, Z_mn, initial_config)
            
        
        
        reg = 0.01 * (np.sum(x**2))
        
        return val + reg


    
    
    
    def min_aspect_ratio_constraint(x):
        R_mn, Z_mn = reshape_coeffs(x)
        violations = constraints.check_constraints(R_mn, Z_mn, {'min_aspect_ratio': 6.0})
        
        from src.physics import geometry
        ar = geometry.calculate_aspect_ratio(R_mn, Z_mn)
        return ar - 6.0 
        
    cons = (
        {'type': 'ineq', 'fun': min_aspect_ratio_constraint},
    )
    
    
    history = []
    def callback(x):
        val = loss_function(x)
        history.append(val)
    
    
    print(f"Initial Loss: {loss_function(x0)}")
    
    
    res = minimize(
        loss_function, 
        x0, 
        method='SLSQP', 
        constraints=cons,
        callback=callback,
        options={'maxiter': max_iter, 'disp': True}
    )
    
    print(f"Optimization finished. Success: {res.success}, Message: {res.message}")
    print(f"Final Loss: {res.fun}")
    
    
    R_opt, Z_opt = reshape_coeffs(res.x)
    optimized_config = initial_config.copy()
    optimized_config['boundary.r_cos'] = R_opt.tolist()
    optimized_config['boundary.z_sin'] = Z_opt.tolist()
    
    
    optimized_config['optimization_history'] = history
    
    return optimized_config
