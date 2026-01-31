import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from src.io.loader import load_constellaration_dataset


class StellaratorDataset(Dataset):
    """
    Dataset PyTorch for ConStellaration.
    Input: 1D Tensor of flattened Fourier coefficients (R_mn, Z_mn).
    Target: Scalar (metrics.qi).
    """
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.scaler_X = StandardScaler()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        sample = self.X[idx]
        tensor = [torch.tensor(row, dtype=torch.float32) for row in sample]
        sample = torch.stack(tensor)
        # sample = self.scaler_X.fit_transform(sample)
        # sample = torch.tensor(sample, dtype=torch.float32)
        target = self.y[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample, target



class StellaratorDataModule:
    """
    DataModule for ConStellaration.
    """
    target_cols = [
        ('metrics.aspect_ratio','aspect_ratio'),
        ('metrics.edge_rotational_transform_over_n_field_periods','iota_edge'),
        ('metrics.edge_magnetic_mirror_ratio','mirror_ratio'),
        ('metrics.qi','qi'),
        ('metrics.vacuum_well','w_mhd'),
        ('metrics.flux_compression_in_regions_of_bad_curvature','flux_compression'),
        ('metrics.max_elongation','max_elongation'),
        ('metrics.minimum_normalized_magnetic_gradient_scale_length','min_mag_grad_length'),
    ]

    def __init__(self, batch_size=64, seed=42):
        self.batch_size = batch_size
        self.seed = seed
        pass

    def _extract_coefficients(self, row):
        """
        Extracts and flattens the Fourier coefficients (lists of lists) into a single 1D vector.
        """
        keys = ['boundary.r_cos', 'boundary.z_sin']
        flat_features = []
        
        for k in keys:
            val = row.get(k)
            if val is None:
                raise ValueError(f"Key {k} missing.")
            features_val = np.array(val).ravel()
            flat_features.append(features_val)
        return np.concatenate(flat_features)

    def load_and_process_data(self):
        print("Downloading dataset...")
        
        hf_dataset = load_constellaration_dataset(split="train")
        df = hf_dataset.to_pandas()
        
        print(f"Dataset loaded: {len(df)} samples.")
        
        #target_col = 'metrics.qi'

        
        for target_col in self.target_cols:
            if target_col[0] not in df.columns:
                if 'metrics' in df.columns and isinstance(df.iloc[0]['metrics'], dict):
                    df[target_col[0]] = df['metrics'].apply(lambda x: x.get(target_col[0]) if x else np.nan)
        
        n_before = len(df)
        for target_col in self.target_cols:
            df = df.dropna(subset=[target_col[0]])
        print(f"{n_before - len(df)} invalid samples (NaN). Total: {len(df)}")

        X_list = []
        y_list = []

        for _, row in tqdm(df.iterrows(), desc="Extracting features", total=len(df)):
            try:
                comps = self._extract_coefficients(row)
                X_list.append(comps)
                y_dict = {}
                for target_col in self.target_cols:
                    if target_col[1] == 'w_mhd':
                        y_dict[target_col[1]] = row[target_col[0]] / 4.0
                    else:
                        y_dict[target_col[1]] = row[target_col[0]]
                y_list.append(y_dict)
            except Exception as e:
                print(f"Error processing row: {e}")
                continue


        X = np.array(X_list)
        return X, y_list

    def setup(self):
        X, y = self.load_and_process_data()
        # subset_size = min(10000, len(X))
        # indices = np.random.choice(len(X), subset_size, replace=False)
        # print(X.shape)
        # self.scaler_X.fit(X[indices])
        # X_scaled = self.scaler_X.transform(X)
        #X_scaled = self.scaler_X.fit_transform(X)
        self.train_ds = StellaratorDataset(X, y)
        
        print(f"Training Set Size: {len(self.train_ds)}")

    def get_loader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )
    
    def get_dim_input(self):
        return self.train_ds.X[0].shape