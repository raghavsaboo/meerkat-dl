from __future__ import annotations

from typing import List

import numpy as np
from mdl.autodiff.operations import ParameterOperation
from mdl.tensor import Parameter
from mdl.tensor import Tensor


class Linear(ParameterOperation):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weights = Parameter(np.random.randn(input_size, output_size))
        self.bias = Parameter(np.zeros((1, output_size)))

    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError("Linear operation expects 1 input tensor.")
        input_tensor = input_tensors[0]

        output_data = (
            np.dot(input_tensor.data, self.weights.data) + self.bias.data
        )
        output_tensor = Tensor(output_data, requires_grad=self.requires_grad)

        parents = [input_tensor, self.weights, self.bias]
        # Add edges between output tensor and parameter tensors
        self.global_dc_graph.add_edge(output_tensor, parents)

        output_tensor.backward_fn = self.backward
        output_tensor.parent_broadcast_shape = self.input_broadcast_shape(
            parents
        )

        return output_tensor

    def backward(self, input_tensors: List[Tensor]) -> None:
        if len(input_tensors) != 1:
            raise ValueError(
                "Linear operation backward expects 1 input tensor."
            )
        input_tensor = input_tensors[0]

        # Compute gradients with respect to parameters
        self.weights.grad_fn = lambda output_grad: np.dot(
            input_tensor.data.T, output_grad
        )
        self.bias.grad_fn = lambda output_grad: np.sum(
            output_grad, axis=0, keepdims=True
        )
