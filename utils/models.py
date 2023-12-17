import sys
sys.path.append('..')
import torch
from torch import nn
from torch.nn import functional as F
from utils.layers import *
# Create a BaseBatchEnsemble Class which take in layers and units as arguments
class BatchEnsemble(nn.Module):
    def __init__(self,
                 ensemble_size: int,
                 input_shape: int, 
                 hidden_layers: int, 
                 hidden_units: int):
        super().__init__()

        # Create the first layer
        layers = [BatchLinear(ensemble_size=ensemble_size,
                              in_features=input_shape,
                              out_features=hidden_units,
                              ), 
                  nn.ReLU()]

        # Add the hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(BatchLinear(ensemble_size=ensemble_size,
                                      in_features=hidden_units,
                                      out_features=hidden_units))
            layers.append(nn.ReLU())

        # Add the output layer
        layers.append(BatchLinear(ensemble_size=ensemble_size,
                                  in_features=hidden_units,
                                  out_features=hidden_units))

        # Create the layer stack
        self.layer_stack = nn.Sequential(*layers)

        # Create mean layer
        self.mean_layer = BatchLinear(ensemble_size=ensemble_size,
                                     in_features=hidden_units,
                                     out_features=1)
        # Create variance layer
        self.var_layer = BatchLinear(ensemble_size=ensemble_size,
                                     in_features=hidden_units,
                                     out_features=1)

    def forward(self, x):
        # Returns hidden units
        z = self.layer_stack(x)

        # Send the hidden units to mean and variance layer
        mean = self.mean_layer(z)
        # Ensure that variance is >= 0
        var = F.softplus(self.var_layer(z))
        return mean, var

class AnchoredBatchEnsemble(nn.Module):
    def __init__(self,
                 ensemble_size: int,
                 input_shape: int, 
                 hidden_layers: int, 
                 hidden_units: int,
                 dropout_prob: float=0.0,
                 use_first_layer_init: bool=True,
                 old_bias: bool=False,
                 mode: str='fan_in',
                 expand: bool=True,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        super().__init__()
        # If dropout probability is larger than 0 we are using dropout regularization
        if dropout_prob > 0.0:
            layers = [AnchoredBatch(ensemble_size=ensemble_size,
                              in_features=input_shape,
                              out_features=hidden_units,
                              device=device,
                              is_first_layer=use_first_layer_init,
                              old_bias= old_bias,
                              mode=mode,
                              expand=expand
                              ), 
                  nn.ReLU(),
                  nn.Dropout(p=dropout_prob)]
            
            for _ in range(hidden_layers - 1):
                layers.append(AnchoredBatch(ensemble_size=ensemble_size,
                                        in_features=hidden_units,
                                        out_features=hidden_units,
                                        device=device,
                                        old_bias=old_bias,
                                        mode=mode,
                                        expand=expand))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_prob))
            
        else:
            # Create the first layer
            layers = [AnchoredBatch(ensemble_size=ensemble_size,
                                in_features=input_shape,
                                out_features=hidden_units,
                                device=device,
                                is_first_layer=use_first_layer_init,
                                old_bias= old_bias,
                                mode=mode,
                                expand=expand
                                ), 
                    nn.ReLU()]

            # Add the hidden layers
            for _ in range(hidden_layers - 1):
                layers.append(AnchoredBatch(ensemble_size=ensemble_size,
                                        in_features=hidden_units,
                                        out_features=hidden_units,
                                        device=device,
                                        old_bias=old_bias,
                                        mode=mode,
                                        expand=expand))
                layers.append(nn.ReLU())

        # Add the output layer
        layers.append(AnchoredBatch(ensemble_size=ensemble_size,
                                  in_features=hidden_units,
                                  out_features=hidden_units,
                                  device=device,
                                  old_bias=old_bias,
                                  mode=mode,
                                  expand=expand))

        # Create the layer stack
        self.layer_stack = nn.Sequential(*layers)

        # Create mean layer
        self.mean_layer = BatchLinear(ensemble_size=ensemble_size,
                                      in_features=hidden_units,
                                      out_features=1,
                                      device=device,
                                      old_bias=old_bias,
                                      mode=mode,
                                      expand=expand)
        # Create variance layer
        self.var_layer = BatchLinear(ensemble_size=ensemble_size,
                                      in_features=hidden_units,
                                      out_features=1,
                                      device=device,
                                      old_bias=old_bias,
                                      mode=mode,
                                      expand=expand)

    def forward(self, x):
        # Returns hidden units
        z = self.layer_stack(x)
        # Send the hidden units to mean and variance layer
        mean = self.mean_layer(z)
        # Ensure that variance is >= 0
        var = F.softplus(self.var_layer(z))
        return mean, var
    

class KaimingNN(nn.Module):
    def __init__(self,
                 input_shape: int, 
                 hidden_layers: int, 
                 hidden_units: int):
        super().__init__()

        # Create the first layer
        layers = [Kaiming_Linear(in_features=input_shape,
                              out_features=hidden_units,
                              ), 
                  nn.ReLU()]

        # Add the hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(Kaiming_Linear(in_features=hidden_units,
                                      out_features=hidden_units))
            layers.append(nn.ReLU())

        # Add the output layer
        layers.append(Kaiming_Linear(in_features=hidden_units,
                                  out_features=hidden_units))

        # Create the layer stack
        self.layer_stack = nn.Sequential(*layers)

        # Create mean layer
        self.mean_layer = Kaiming_Linear(in_features=hidden_units,
                                     out_features=1)
        # Create variance layer
        self.var_layer = Kaiming_Linear(
                                     in_features=hidden_units,
                                     out_features=1)
    def forward(self, x):
        # Returns hidden units
        z = self.layer_stack(x)
        # Send the hidden units to mean and variance layer
        mean = self.mean_layer(z)
        # Ensure that variance is >= 0
        var = F.softplus(self.var_layer(z))
        return mean, var