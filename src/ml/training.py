import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_model(model, train_loader, target, epochs=10, lr=1e-3, device="cpu"):
    """
    training loop for regression
    """
    model = model.to(device)
    model.train() 
    
    criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    print(f"Starting training on device: {device}")
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
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
        loss_history.append(avg_loss)
        
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        
        print(f"\tEpoch {epoch+1} Summary | Avg Loss: {avg_loss:.6f} | LR: {current_lr:.1e}")

    print("Training completed.")
    return model, loss_history