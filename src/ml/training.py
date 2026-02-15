from sklearn.metrics import r2_score
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
import numpy as np
from src.utils.config import getLogger


def train_model(model, train_loader, val_loader, target, epochs=10, lr=1e-3, device="cpu"):
    """
    training loop for regression
    """
    model = model.to(device)
    model.train() 
    
    criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    print(f"Starting training on device: {device}")
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    logger = getLogger("production")
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")
        
        for batch_X, batch_y in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y[target].to(device).float()
            
            batch_y = batch_y.view(-1, 1)
            optimizer.zero_grad()
            
            
            preds = model(batch_X)
            
            
            loss = criterion(preds, batch_y)
            
            
            loss.backward()
            optimizer.step()
            
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        avg_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_loss)

        model.eval()
        val_loss_accum = 0.0
        val_preds = []
        val_targets = []

        pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}")
        with torch.no_grad():
            for X_val, Y_val in pbar:
                batch_X, batch_y = X_val.to(device), Y_val[target].to(device).float()
            
                batch_y = batch_y.view(-1, 1)
                
                val_pred = model(batch_X)
                loss = criterion(val_pred, batch_y)
                
                val_loss_accum += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.6f}"})
                
                val_preds.append(val_pred.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

        avg_val_loss = val_loss_accum / len(val_loader)
        
        history['val_loss'].append(avg_val_loss)
        all_val_preds = np.vstack(val_preds)
        all_val_targets = np.vstack(val_targets)
        
        val_accuracy = r2_score(all_val_targets, all_val_preds)
        history['val_accuracy'].append(val_accuracy)
        
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        print(f"\tEpoch {epoch+1} Summary | Avg Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Accuracy: {val_accuracy:.6f} | LR: {current_lr:.1e}")
        logger.info(f"\tEpoch {epoch+1} Summary | Avg Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Accuracy: {val_accuracy:.6f} | LR: {current_lr:.1e}")

    print("Training completed.")
    return model, history