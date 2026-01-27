import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class StellaratorSurrogate(nn.Module):
    def __init__(self, input_shape, hidden_dims=[512, 512, 256]):
        """
        Sequential Neural Network for QI regression
        
        Args:
            input_shape (int): shape of the flattened coefficients vector
            hidden_dims (list): list of hidden layer dimensions
        """
        super(StellaratorSurrogate, self).__init__()
        self.flat_dim = 1
        for d in input_shape:
            self.flat_dim *= d
        layers = []
        layers.append(nn.Flatten())
        
        in_channels = self.flat_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_channels, h_dim))
            layers.append(nn.GELU())  
            layers.append(nn.BatchNorm1d(h_dim)) 
            layers.append(nn.Dropout(0.1)) 
            in_channels = h_dim
        
        layers.append(nn.Linear(in_channels, 1))
        
        self.net = nn.Sequential(*layers)
    

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, epochs=10, lr=1e-3, device="cpu"):
    """
    training loop for regression
    """
    model = model.to(device)
    model.train() 
    
    criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    print(f"Starting training on device: {device}")
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_X, batch_y in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            
            optimizer.zero_grad()
            
            
            preds = model(batch_X)
            
            
            loss = criterion(preds, batch_y)
            
            
            loss.backward()
            optimizer.step()
            
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        avg_loss = total_loss / len(train_loader)
        
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        
        print(f"\t📉 Epoch {epoch+1} Summary | Avg Loss: {avg_loss:.6f} | LR: {current_lr:.1e}")

    print("Training completed.")
    return model
