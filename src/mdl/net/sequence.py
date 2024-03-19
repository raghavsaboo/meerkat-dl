from __future__ import annotations

from typing import List

import numpy as np
from mdl.net.layer import Layer
from mdl.tensor import Parameter
from mdl.tensor import Tensor
from mdl.net.activation import ActivationFunction
from mdl.net.activation import Activation
from mdl.autodiff.operations import stack

class RNNCell(Layer):
    def __init__(self, input_size: int, hidden_size: int, use_bias: bool = True, activation: Activation = Activation.TANH):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

        # Initialize weights and biases
        self.W_ih = Parameter(np.random.randn(input_size, hidden_size) * 0.01)
        self.W_hh = Parameter(np.random.randn(hidden_size, hidden_size) * 0.01)
        if use_bias:
            self.b_ih = Parameter(np.zeros((1, hidden_size)))
            self.b_hh = Parameter(np.zeros((1, hidden_size)))
        else:
            self.b_ih = self.b_hh = None
        
        # Activation function
        self.activation = ActivationFunction.get(activation)

    def forward(self, input_tensor: Tensor, hidden_state: Tensor) -> Tensor:
        # Using MatMul operation for matrix multiplication
        input_contrib = input_tensor @ self.W_ih
        hidden_contrib = hidden_state @ self.W_hh

        if self.use_bias:
            # Using Add operation for adding bias
            input_contrib += self.b_ih
            hidden_contrib += self.b_hh

        # Combining input and hidden contributions
        combined = input_contrib + hidden_contrib
        
        new_hidden_state = self.activation(combined)

        return new_hidden_state
    

class RNNLayer(Layer):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.rnn_cell = RNNCell(input_size, hidden_size)  # Assuming RNNCell also follows the framework's conventions
        self.hidden_size = hidden_size

    def forward(self, input_tensor: Tensor) -> List[Tensor]:
        """
        Processes an input sequence through the RNNLayer, inheriting from the Layer class.

        :param input_tensor: A Tensor of shape (sequence_length, batch_size, input_feature_size).
        :return: A list of hidden state Tensors for each time step.
        """
        sequence_length, batch_size, _ = input_tensor.shape
        hidden_states = []

        # Initialize the hidden state as zeros
        hidden_state = Tensor(np.zeros((batch_size, self.hidden_size)), requires_grad=True)

        for t in range(sequence_length):
            current_input = input_tensor[t, :, :]  # Using enhanced slicing for tensors
            hidden_state = self.rnn_cell.forward(current_input, hidden_state)
            hidden_states.append(hidden_state)
            
        hidden_states = stack(hidden_states, axis=0)

        return hidden_states