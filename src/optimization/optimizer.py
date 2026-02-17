from src.physics import geometry
from src.optimization import objectives
import numpy as np
from scipy.optimize import minimize
from src.ml.ensamble import StellaratorEnsemble
import torch

WEIGHTS = {
    'qi': 1.0,
    'well': 10.0,
    'mirror': 2.0,
    'iota': 5.0,
    'reg': 0.0001,
    'aspect': 0.0
}

SCALES = {
    'qi': 0.05,
    'well': 0.02,
    'iota': 0.4,
    'aspect': 8.0,
    'mirror': 0.2 
}

def create_targets(r_mn, z_mn, n_field_periods, target_iota=0.42):

    targets = {}
    
    targets['qi'] = 0.0
    targets['well'] = 0.0
    targets['iota'] = target_iota 
    targets['volume'] = geometry.calculate_volume(r_mn, z_mn, n_field_periods)
    
    targets['aspect_ratio'] = geometry.calculate_aspect_ratio(r_mn, z_mn)

        
    return targets

def spectral_regularization(x_tensor):
    """
    Penalizes high-frequency modes to keep the surface smooth.
    """
    
    n_coeffs = x_tensor.shape[0] // 2
    
    indices = torch.arange(n_coeffs, dtype=torch.float32, device=x_tensor.device)
    weights = (1.0 + indices / n_coeffs * 9.0) ** 2
    
    r_coeffs = x_tensor[:n_coeffs]
    z_coeffs = x_tensor[n_coeffs:]
    
    reg_loss = torch.sum(weights * (r_coeffs**2)) + torch.sum(weights * (z_coeffs**2))
    return reg_loss


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
    n_field_periods = initial_config['boundary.n_field_periods']
    
    shape_R = R_mn_init.shape
    shape_Z = Z_mn_init.shape
    
    target_volume = geometry.calculate_volume(R_mn_init, Z_mn_init)
    x0 = np.concatenate([R_mn_init.flatten(), Z_mn_init.flatten()])

    
    models = StellaratorEnsemble("configs/conf.yaml", "configs/model_struct.json")
    models.load_models()

    targets = create_targets(R_mn_init, Z_mn_init, n_field_periods, target_iota=0.42)
    
    def reshape_coeffs(x):
        split = R_mn_init.size
        R_flat = x[:split]
        Z_flat = x[split:]
        return R_flat.reshape(shape_R), Z_flat.reshape(shape_Z)
    
    
    def loss_function(x):
        R_mn, Z_mn = reshape_coeffs(x)
        x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True, device="cpu")
        
        
        if problem_type == "GeoFusion-nn":
            metrics_val = objectives.calculate_geo_fusion_nn(R_mn, Z_mn, models)

            loss = 0.0
        
            loss += WEIGHTS['qi'] * (metrics_val['qi'] / SCALES['qi'])**2
                
        
            iota_val = metrics_val['iota_edge']
            loss += WEIGHTS['iota'] * ((iota_val - targets['iota']) / SCALES['iota'])**2
                
            
            well_val = metrics_val['w_mhd']
            loss += WEIGHTS['well'] * (torch.relu(well_val) / SCALES['well'])**2
                
            
            mr_val = metrics_val['mirror_ratio']
            loss += WEIGHTS['mirror'] * (mr_val / SCALES['mirror'])**2
                
            
            reg_loss = spectral_regularization(x_tensor)
            loss += WEIGHTS['reg'] * reg_loss
            
            loss.backward()
            
            loss_val = loss.item()
            grad_val = x_tensor.grad.detach().cpu().numpy().astype(np.float64)
            
            print(f"Loss: {loss_val:.6f} | Reg: {reg_loss.item():.2e}", end="\033[K\r")
            
            return loss_val, grad_val



        elif problem_type == "simple-to-build":
            
            val = objectives.calculate_coil_simplicity(R_mn, Z_mn, initial_config)
        else:
            
            val = objectives.calculate_mhd_stability(R_mn, Z_mn, initial_config)
            
        
        
        reg = 0.01 * (np.sum(x**2))
        
        return val + reg


    

    def volume_cons(x):
        r, z = reshape_coeffs(x)
        current_vol = geometry.calculate_volume(r, z, n_field_periods)
        return (current_vol - targets['volume']) / targets['volume']
    
    def ar_cons(x):
        r, z = reshape_coeffs(x)
        current_ar = geometry.calculate_aspect_ratio(r, z)
        return current_ar - 6.0
        
    cons= (
        {'type': 'eq',   'fun': volume_cons},
        {'type': 'ineq', 'fun': ar_cons}
    )
    
    
    history = []
    def callback(x):
        val = loss_function(x)
        history.append(val[0])
        return val
    
    
    
    res = minimize(
        callback,
        x0,
        method='SLSQP',
        jac=True,
        constraints=cons,
        options={'maxiter': max_iter, 'disp': True, 'ftol': 1e-6}
    )
    
    
    
    R_opt, Z_opt = reshape_coeffs(res.x)
    optimized_config = initial_config.copy()
    optimized_config['boundary.r_cos'] = R_opt.tolist()
    optimized_config['boundary.z_sin'] = Z_opt.tolist()
    
    
    optimized_config['optimization_history'] = history
    
    return optimized_config
