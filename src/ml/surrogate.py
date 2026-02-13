import torch.nn as nn

class StellaratorSurrogate(nn.Module):
    def __init__(self, input_shape, hidden_dims, activation='GELU', structure=None):
        """
        Sequential Neural Network for surrogate modeling of stellarator metrics.
        
        Args:
            input_shape (int): shape of the flattened coefficients vector
            hidden_dims (list): list of hidden layer dimensions
            activation (str): activation function to use ('GELU' or 'ReLU')
            structure (list): optional JSON-like structure for dynamic model creation
        """
        super(StellaratorSurrogate, self).__init__()
        if structure is not None:
            self.model = JSONModel(input_shape, structure)
            
        else:
            if hidden_dims is None:
                hidden_dims = [512, 256, 128] 
                
            layers = []
            in_dim = input_shape
            self.flatten = nn.Flatten()
            
            act_fn = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
            
            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(act_fn)
                in_dim = h_dim
            
            layers.append(nn.Linear(in_dim, 1))
            
            self.model = nn.Sequential(
                self.flatten,
                *layers
            )

    def forward(self, x):
        
        return self.model(x)


class JSONModel(nn.Module):
    def __init__(self, input_dim, json_struct):
        super().__init__()
        
        self.flatten = nn.Flatten()
        layers = []
        in_features = input_dim
        
        for item in json_struct:
            item_type = item.get("type")
            
            if item_type == "layer":
                l_type = item.get("type_layer")
                
                if l_type == "linear":
                    out_features = item.get("out_features")
                    layers.append(nn.Linear(in_features, out_features))
                    in_features = out_features
                    
                elif l_type == "batch_norm1d":
                    features = item.get("features")
                    layers.append(nn.BatchNorm1d(features))
                    
                elif l_type == "dropout":
                    p = item.get("p", 0.1)
                    layers.append(nn.Dropout(p))
            
            elif item_type == "activation":
                act_name = item.get("name").lower()
                if act_name == "gelu":
                    layers.append(nn.GELU())
                elif act_name == "relu":
                    layers.append(nn.ReLU())
                    
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)