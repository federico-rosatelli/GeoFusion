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
        
        layers = []
        
        self.start_with_conv = False
        self.first_layer_in_channels = None
        
        if json_struct and json_struct[0].get("type_layer") == "conv1d":
            self.start_with_conv = True
            c_in = json_struct[0].get("in_channels")
            self.first_layer_in_channels = c_in
            current_shape = (c_in, input_dim)
        else:
            current_shape = (input_dim,)
        
        for item in json_struct:
            item_type = item.get("type")
            
            if item_type == "layer":
                l_type = item.get("type_layer")
                
                if l_type == "conv1d":
                    in_c = item.get("in_channels")
                    out_c = item.get("out_channels")
                    k = item.get("kernel_size")
                    p = item.get("padding", 0)
                    s = item.get("stride", 1)
                    d = item.get("dilation", 1)
                    
                    layers.append(nn.Conv1d(
                        in_channels=in_c, 
                        out_channels=out_c, 
                        kernel_size=k, 
                        padding=p, 
                        stride=s, 
                        dilation=d
                    ))
                    
                    if len(current_shape) == 2:
                        l_in = current_shape[1]
                        l_out = (l_in + 2*p - d*(k-1) - 1)//s + 1
                        current_shape = (out_c, l_out)
                    
                elif l_type == "max_pool1d":
                    k = item.get("kernel_size")
                    s = item.get("stride", k)
                    p = item.get("padding", 0)
                    d = item.get("dilation", 1)
                    
                    layers.append(nn.MaxPool1d(kernel_size=k, stride=s, padding=p, dilation=d))
                    
                    if len(current_shape) == 2:
                        l_in = current_shape[1]
                        l_out = (l_in + 2*p - d*(k-1) - 1)//s + 1
                        current_shape = (current_shape[0], l_out)
                        
                elif l_type == "flatten":
                    layers.append(nn.Flatten())
                    if len(current_shape) == 2:
                        current_shape = (current_shape[0] * current_shape[1],)
                        
                elif l_type == "linear":
                    out_features = item.get("out_features")
                    in_features = current_shape[0]
                    
                    layers.append(nn.Linear(in_features, out_features))
                    current_shape = (out_features,)
                    
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
        if self.start_with_conv and self.first_layer_in_channels == 1:
            if x.dim() > 2:
                 x = x.flatten(start_dim=1)
            
            if x.dim() == 2:
                x = x.unsqueeze(1)
            
        return self.net(x)