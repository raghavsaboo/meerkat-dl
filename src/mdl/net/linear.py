from __future__ import annotations

import numpy as np
from mdl.net.components import Layer
from mdl.tensor import Parameter
from mdl.tensor import Tensor


class LinearLayer(Layer):
    def __init__(
        self, input_size: int, output_size: int, use_bias: bool = True
    ):
        super(Layer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias

        # Initialize weights
        self.weights = Parameter(
            np.random.randn(input_size, output_size) * 0.01
        )

        # Initialize bias if needed
        if self.use_bias:
            self.bias: Parameter = Parameter(np.zeros((1, output_size)))

    def forward(self, input_tensor: Tensor) -> Tensor:
        # Compute the linear transformation using the @ operator
        weighted_input = input_tensor @ self.weights

        # Add bias if applicable using the + operator
        if self.use_bias:
            weighted_input += self.bias

        print(weighted_input.shape)

        return weighted_input
