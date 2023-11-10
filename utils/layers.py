import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn import init
import math

class BatchLinear(Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        ensemble_size: size of ensemble
        in_featuers: input dimension
        out_features: output dimension
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> model = BatchLinear(4, 20, 30)
        >>> input = torch.randn(128, 20).repeat(4, 1)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([4, 128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    ensemble_size: int
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self,ensemble_size: int , in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        # Ensemble Size
        self.ensemble_size = ensemble_size
        # Input Dimension
        self.in_features = in_features
        # Output Dimension
        self.out_features = out_features

        # Initiate the Shared Weight Matrix
        self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        
        # Initiate the Fast Weights
        self.r = Parameter(torch.empty((ensemble_size, in_features, 1), **factory_kwargs)) 
        self.s = Parameter(torch.empty((ensemble_size, out_features, 1), **factory_kwargs)) 
        if bias:
            self.bias = Parameter(torch.empty((ensemble_size, out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.r, a =math.sqrt(5))
        init.kaiming_uniform_(self.s, a =math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # Calculate the batch size
        batch_size = input.shape[0]//self.ensemble_size

        # Initiate a matrix Y were the results will be stored, will have dimensions B*M x n
        Y = torch.zeros(batch_size*self.ensemble_size, self.out_features, device=input.device)

        # Iterate through the data of each ensemble member
        for i, ri, si in zip(range(self.ensemble_size), self.r, self.s):
            # Define matrices R and S whose rows consist of vectors ri and si for all points in mini-batch
            R = ri.flatten().repeat(batch_size,1)
            S = si.flatten().repeat(batch_size,1)

            # Define indeces for accessing each ensemble member's mini batch
            start_idx = i * batch_size
            end_idx = (i+1) * batch_size

            # Access corresponding mini batch of ensemble member i
            X = input[start_idx:end_idx]

            # Compute the forward pass using Equation 5 of BatchEnsemble
            Y[start_idx:end_idx] = torch.mm((X*R), self.weight) * S
            if self.bias is not None:
                Y[start_idx:end_idx] += self.bias[i]
        return Y
        
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'