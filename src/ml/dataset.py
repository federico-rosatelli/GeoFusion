import numpy as np
from src.io.loader import load_constellaration_dataset

class ConstellarationDataset:
    def __init__(self, verbose=False):
        self.data = load_constellaration_dataset()
        self.verbose = verbose
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        R_mn = np.array(self.data[idx]['boundary.r_cos']).flatten()
        Z_mn = np.array(self.data[idx]['boundary.z_sin']).flatten()
        X = np.concatenate([R_mn, Z_mn])
        print(X)

        Y = {
            'aspect_ratio': self.data[idx]['metrics.aspect_ratio'],
            'iota_edge': self.data[idx]['metrics.edge_rotational_transform_over_n_field_periods'],
            'mirror_ratio': self.data[idx]['metrics.edge_magnetic_mirror_ratio'],
            'qi_residual': self.data[idx]['metrics.qi'],
            'w_mhd': self.data[idx]['metrics.vacuum_well'],
            'chi_nabla_r': self.data[idx]['metrics.flux_compression_in_regions_of_bad_curvature'],
            'max_elongation': self.data[idx]['metrics.max_elongation'],
            'e_L_nablaB': self.data[idx]['metrics.minimum_normalized_magnetic_gradient_scale_length']
        }
        return X, Y
    
    def prepare_training_data(self):
        
        X = []
        Y = {}
        
        for config in self.data:
            R_mn = np.array(config['boundary.r_cos']).flatten()
            Z_mn = np.array(config['boundary.z_sin']).flatten()
            X.append(np.concatenate([R_mn, Z_mn]))
            
            y_sample = {
                'aspect_ratio': config['metrics.aspect_ratio'],
                'iota_edge': config['metrics.edge_rotational_transform_over_n_field_periods'],
                'mirror_ratio': config['metrics.edge_magnetic_mirror_ratio'],
                'qi_residual': config['metrics.qi'],
                'w_mhd': config['metrics.vacuum_well'],
                'chi_nabla_r': config['metrics.flux_compression_in_regions_of_bad_curvature'],
                'max_elongation': config['metrics.max_elongation'],
                'e_L_nablaB': config['metrics.minimum_normalized_magnetic_gradient_scale_length']
            }
            
            for key, val in y_sample.items():
                Y.setdefault(key, []).append(val)
            if self.verbose:
                print(f"Processed {len(X)/len(self.data)*100:.2f}%\r", end="")
            
            
                
        return np.array(X), {k: np.array(v) for k, v in Y.items()}