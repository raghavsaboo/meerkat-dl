from mdl.autodiff.operations import ParameterOperation
from mdl.tensor import Tensor
import numpy as np
from typing import List

class Sigmoid(ParameterOperation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(1 / (1 + np.exp(-a.data)), self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad * (1 / (1 + np.exp(-a.data))) * (1 - 1 / (1 + np.exp(-a.data)))

class ReLU(ParameterOperation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(np.maximum(0, a.data), self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad * (a.data > 0).astype(float)

class Tanh(ParameterOperation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(np.tanh(a.data), self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad * (1 - np.tanh(a.data)**2)