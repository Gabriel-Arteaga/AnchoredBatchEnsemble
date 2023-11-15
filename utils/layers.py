import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from torch.nn import functional as F

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
    

class ModifiedLinear(Module):
    """
    A modified linear layer which uses Kaiming Normal Initialization. 
    Stores mean and standard deviation of the initalization parameters.


    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # Prior mean
        self.mean = 0
        # Prior standard deviation, will be calculatedin reset_parameters()
        self.std = 0

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        #init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # We use Kaiming Normal instead of Kaiming Uniform initialization
        init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        
        # We calculate the standard deviation, which is the same std deviation used to initalize the weights
        fan = init._calculate_correct_fan(self.weight, mode='fan_out')
        gain = init.calculate_gain(nonlinearity='relu')
        self.std = gain / math.sqrt(fan)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class AnchoredBatch(Module):
    """
    Args:
        ensemble_size: size of ensemble
        in_featuers: input dimension
        out_features: output dimension
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    """
    __constants__ = ['ensemble_size', 'in_features', 'out_features']
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

        # Prior mean
        self.mean = 0

        # Prior standard deviation, will be calculatedin reset_parameters()
        self.std = 0

        # Initalize the anchor points for each ensemble member
        self.anchors = torch.empty((self.ensemble_size, self.in_features, self.out_features), **factory_kwargs)

        # Initiate the Shared Weight Matrix
        self.weight = torch.empty((in_features, out_features), **factory_kwargs) # Train shared as well
        
        # Initiate the Fast Weights
        self.r = Parameter(torch.empty((ensemble_size, in_features, 1), **factory_kwargs)) 
        self.s = Parameter(torch.empty((ensemble_size, out_features, 1), **factory_kwargs)) 
        if bias:
            self.bias = Parameter(torch.empty((ensemble_size, out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.set_anchors()

    def reset_parameters(self) -> None:
        """
        Instead of using Kaiming Uniform initailization which is default for Linear module we instead use Kaiming Normal.
        This allows us to use the mean and standard deviation of the weight initialization as our prior for our anchored ensemble.  
        """
        init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

        # Calculate the standard deviation 
        fan = init._calculate_correct_fan(self.weight, mode='fan_out')
        gain = init.calculate_gain(nonlinearity='relu')
        self.std = gain / math.sqrt(fan)
        # Initialize r and s vectors with same parameters as the weight matrix
        with torch.no_grad():
            self.r.normal_(0, self.std)
            self.s.normal_(0, self.std)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        
    def set_anchors(self):
        # Draw anchor points for each ensemble member using the prior mean and std
        for ensemble_index in range(self.ensemble_size):
            self.anchors[ensemble_index] = torch.normal(mean=self.mean, std = self.std, size =(self.in_features, self.out_features))

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
    
    def get_reg_term(self, N: int, data_noise: float):
        """
        Args:
        N: number of data points
        data_noise: noise of data
        """
        # Initalize constants
        tau = data_noise/self.std 
        normalization_term = 1/N 

        reg_term = 0
        # Iterate through each ensemble member and compute the regularization
        for i in range(self.ensemble_size):
            reg_term += normalization_term*tau*torch.mul(self.weight*(self.r[i]@self.s[i].T)-self.anchors[i],self.weight*(self.r[i]@self.s[i].T)-self.anchors[i]).sum()
        return reg_term

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'