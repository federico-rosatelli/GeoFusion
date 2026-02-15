import os
import csv
import pandas as pd
import torch
from src.ml.surrogate import StellaratorSurrogate
from src.ml.training import train_model
from src.utils.config import load_config
from src.visualization.plotting import plot_loss

class StellaratorEnsemble:
    """Ensemble of surrogate models for uncertainty estimation"""
    
    def __init__(self, config_yaml_path:str, config_json_path:str):

        self.models = {}
        self.models_conf, device_name, self.force_retrain = load_config(config_yaml_path, config_json_path)
        self.device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.loss_histories = {}
        
        for metric, cfg in self.models_conf.items():
            input_dim = cfg['input_dim']
            layers = cfg['train']['layers']
            act = cfg['train']['activation']
            structure = cfg.get('structure', None)
            
            model = StellaratorSurrogate(input_dim, layers, act, structure=structure).to(self.device)
            self.models[metric] = model
            self.loss_histories[metric] = {}
            
    
    def train_model(self, metric, train_loader, val_loader=None):
        mdfcfg = self.models_conf[metric]
        save_path = mdfcfg['filepath']
        if not self.force_retrain and os.path.exists(save_path):
            print(f"Model for metric '{metric}' already exists and force_retrain is False. Skipping training.")
            return None
        
        cfg = mdfcfg["train"]
        model = self.models[metric]
        trained_model, history = train_model(model, train_loader, val_loader, metric, cfg["epochs"], lr=float(cfg["learning_rate"]), device=self.device)
        self.models[metric] = trained_model
        self.loss_histories[metric] = history
        return trained_model
    
    def save_model(self, metric):
        model = self.models[metric]
        save_path = self.models_conf[metric]["filepath"]
        if not self.force_retrain and os.path.exists(save_path):
            print(f"Force_retrain is False for metric '{metric}'. Skipping save.")
            return None
        torch.save(model.state_dict(), save_path)
        return save_path
    
    def save_loss_log(self, metric):
        epochs = self.models_conf[metric]["train"]["epochs"]
        model_path = self.models_conf[metric]["filepath"]
        history = self.loss_histories[metric]
        if self.loss_histories[metric] == {}:
            print(f"No loss history available for metric '{metric}'. Skipping loss log save.")
            return None
        base_path, _ = os.path.splitext(model_path)
        log_path = f"{base_path}_loss.csv"
        
        try:
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy'])
                for i in range(epochs):
                    writer.writerow([i + 1, history['train_loss'][i], history['val_loss'][i], history['val_accuracy'][i]])
        except Exception as e:
            print(f"Error saving loss log for {metric}: {e}")
    
    def predict(self, x):

        predictions = {}
        
        with torch.no_grad():
            for metric, model in self.models.items():
                pred = model(x)
                if metric == 'w_mhd':
                    pred = pred * 4.0
                predictions[metric] = pred
        
        return predictions
    
    def load_models(self):
        for metric, cfg in self.models_conf.items():
            model = self.models[metric]
            path = self.models_conf[metric]["filepath"]
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models[metric] = model
        return self.models
    
    def plot_loss(self, metric):
        if self.loss_histories[metric] != []:
            loss_history = self.loss_histories[metric]
            
        else:
            model_path = self.models_conf[metric]["filepath"]
            base_path, _ = os.path.splitext(model_path)
            log_path = f"{base_path}_loss.csv"
            if not os.path.exists(log_path):
                print(f"Log file for metric '{metric}' not found at {log_path}.")
                return None
            df = pd.read_csv(log_path)
            loss_history = df['loss'].tolist()
        
        plot_loss(metric, loss_history)