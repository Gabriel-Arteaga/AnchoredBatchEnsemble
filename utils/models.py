import sys
sys.path.append('..')
import torch
from torch import nn
from torch.nn import functional as F
from utils.layers import BatchLinear
from utils.layers import AnchoredBatch
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
                              out_features=hidden_units), 
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
                 output_shape: int):
        super().__init__()

        # Create the first layer
        layers = [BatchLinear(ensemble_size=ensemble_size,
                              in_features=input_shape,
                              out_features=hidden_units), 
                  nn.ReLU()]