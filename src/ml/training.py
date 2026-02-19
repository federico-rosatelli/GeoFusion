from sklearn.metrics import r2_score
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
import numpy as np
from src.utils.config import getLogger


LOG_TRANSFORM_METRICS = {'qi', 'flux_compression', 'min_mag_grad_length'}

HUBER_DELTA = 0.5

def _apply_transform(y: torch.Tensor, metric: str):
    if metric in LOG_TRANSFORM_METRICS:
        return torch.log(torch.clamp(y, min=1e-9))
    return y


def _inverse_transform(y: torch.Tensor, metric: str):
    if metric in LOG_TRANSFORM_METRICS:
        return torch.exp(y)
    return y



def train_model(model, train_loader, val_loader, target, epochs=10, lr=1e-3, device="cpu"):
    """
    training loop for regression
    """
    model = model.to(device)
    model.train() 

    
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(5, epochs // 4), T_mult=2, eta_min=lr * 1e-3)

    print(f"Starting training on device: {device}")
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    best_state = None
    logger = getLogger("production")
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}")
        
        for batch_X, batch_y in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y[target].to(device).float().view(-1, 1)
            
            batch_y_transformed = _apply_transform(batch_y, target)

            preds = model(batch_X)
            loss = criterion(preds, batch_y_transformed)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                batch_X, batch_y = X_val.to(device), Y_val[target].to(device).float().view(-1, 1)
            
                y_transformed = _apply_transform(batch_y, target)
                
                pred = model(batch_X)
                loss = criterion(pred, y_transformed)
                val_loss_accum += loss.item()
                val_pred = _inverse_transform(pred, target).cpu().numpy()


                pbar.set_postfix({'loss': f"{loss.item():.6f}"})
                
                val_preds.append(val_pred.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())

        avg_val_loss = val_loss_accum / len(val_loader)
        
        history['val_loss'].append(avg_val_loss)
        all_val_preds = np.vstack(val_preds)
        all_val_targets = np.vstack(val_targets)
        
        val_accuracy = r2_score(all_val_targets, all_val_preds)
        history['val_accuracy'].append(val_accuracy)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        print(f"\tEpoch {epoch+1} Summary | Avg Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Accuracy: {val_accuracy:.6f} | LR: {current_lr:.1e}")
        logger.info(f"\tEpoch {epoch+1} Summary | Avg Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Accuracy: {val_accuracy:.6f} | LR: {current_lr:.1e}")


    if best_state is not None:
        model.load_state_dict(best_state)

    print("Training completed.")
    return model, history