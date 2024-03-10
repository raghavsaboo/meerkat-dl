from __future__ import annotations

from typing import List

import numpy as np
from mdl.autodiff.operations import ParameterOperation
from mdl.tensor import Parameter
from mdl.tensor import Tensor

# TODO: Make bias optional
# TODO: Add non-linearity into cell?
class RNNCell(ParameterOperation):
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # use a random uniform distribution for weights and bias
        # sampled from $-/sqrt{k}$ to $/sqrt{k}$
        k = 1./float(hidden_size)
        low = -np.sqrt(k)
        high = np.sqrt(k)
        self.weights_input = Parameter(np.random.uniform(low=low, high=high, size=(input_size, hidden_size)))
        self.weights_hidden = Parameter(np.random.uniform(low=low, high=high, size=(hidden_size, hidden_size)))
        self.bias = Parameter(np.random.uniform(low=low, high=high, size=(1, hidden_size)))
        
    def _forward(self, input_tensors: List[Tensor]) -> List[Tensor]:
        # unpack input tensors where first one is expected to be input
        # at time step t with shape [batch size, input_size]
        # and the second tensor is expected to be the previous hidden state
        # from step t - 1 with shape [batch size, hidden_size]
        if len(input_tensors) != 2:
            raise ValueError("RNNCell operation expects 2 input tensor.")
        input_tensor, prev_hidden_state = input_tensors
        
        if input_tensor.shape[1] != self.input_size:
            raise ValueError(f"RNNCell input shapes mismatch. Expected {self.input_size} but got {input_tensor.shape[1]}")
        
        if prev_hidden_state.shape[1] != self.hidden_size:
            raise ValueError(f"RNNCell hidden state shapes mismatch. Expected {self.hidden_size} but got {prev_hidden_state.shape[1]}")
        
        hidden_state = np.dot(input_tensor.data, self.weights_input.data) + \
            np.dot(prev_hidden_state.data, self.weights_hidden) + \
                self.bias.data
        
                  
        