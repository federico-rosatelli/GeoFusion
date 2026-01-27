from src.ml.surrogate import StellaratorSurrogate, train_model
import torch
from src.ml import *

class StellaratorEnsemble:
    """Ensemble of surrogate models for uncertainty estimation"""
    
    def __init__(self, input_dim, model_struct:dict):

        self.models = {}
        self.model_struct = model_struct
        
        
        for metric in model_struct:

            model = StellaratorSurrogate(input_dim, model_struct[metric]["layers"]).to(DEVICE)
            model.eval()
            self.models[metric] = model
    
    def train_model(self, metric, train_loader, epochs=10, lr=1e-3):
        model = self.models[metric]
        trained_model = train_model(model, train_loader, epochs=epochs, lr=lr, device=DEVICE)
        self.models[metric] = trained_model
    
    def save_model(self, metric, path):
        model = self.models[metric]
        torch.save(model.state_dict(), path)
    
    def predict(self, x):

        predictions = {}
        
        with torch.no_grad():
            for metric, model in self.models.items():
                pred = model(x)
                predictions[metric] = pred
        
        return predictions