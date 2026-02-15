import os
import sys
from src.ml.ensamble import StellaratorEnsemble
from src.io.dataLoader import StellaratorDataModule


class NeuralManager:
    """
    Manager class to orchestrate the training of surrogate models for stellarator optimization.
    This class handles configuration loading, data preparation, model initialization, and training execution.

    Args:
        config_yaml_path (str): Path to the YAML configuration file for training parameters.
        config_json_path (str): Path to the JSON file containing model architecture definitions.
        force_retrain (bool): If True, forces retraining of models even if saved versions exist.
    """
    def __init__(self, config_path="configs/conf.yaml", struct_path="configs/model_struct.json", force_retrain=False):
        self.config_path = config_path
        self.struct_path = struct_path
        self.force_retrain = force_retrain
        
        self._validate_paths()

        self.train_loader, self.val_loader = self._prepare_data()
        self.models = self._init_ensemble()
        
        
    def _validate_paths(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        if not os.path.exists(self.struct_path):
            print(f"Warning: Model structure file not found: {self.struct_path}. Proceeding with None.")
            self.struct_path = None

    def _prepare_data(self):
        try:
            
            dm = StellaratorDataModule() 
            dm.setup()
            train_loader, val_loader = dm.get_loader()
            
            return train_loader, val_loader
        
        except Exception as e:
            print(f"Error preparing data: {e}")
            sys.exit(1)

    def _init_ensemble(self):
        try:
            models = StellaratorEnsemble(self.config_path, self.struct_path)
            
            if self.force_retrain:
                print("Force retrain enabled: All models will be retrained regardless of existing saved versions.")
                models.force_retrain = True
            
            return models
        
        except Exception as e:
            print(f"Error initializing Ensemble: {e}")
            sys.exit(1)

    def train_model(self, metric):
        try:
            print(f"Training model for metric '{metric}'...")
            self.models.train_model(metric, self.train_loader, self.val_loader)
            self.models.save_model(metric)
            #self.models.save_loss_log(metric)
            print(f"Model for metric '{metric}' trained and saved successfully.")
        
        except Exception as e:
            print(f"Error training model for metric '{metric}': {e}")
            sys.exit(1)
    
    def train_all(self):
        for metric in self.models.models.keys():
            self.train_model(metric)
        print("All models trained and saved.")