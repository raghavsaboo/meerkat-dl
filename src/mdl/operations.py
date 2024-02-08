from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import List

import numpy as np
from mdl.autodiff.dcgraph import DCGraph
from mdl.tensor import Tensor


class Operation(ABC):

    global_dc_graph = DCGraph()

    def __call__(self, input_tensors: List[Tensor]):
        self.forward(input_tensors)

    @property
    def requires_grad(self):
        return self.requires_grad

    @requires_grad.setter
    def requires_grad(self, input_tensors: List[Tensor]) -> None:
        self.requires_grad = any(
            [tensor.requires_grad for tensor in input_tensors],
        )

    def validate_input_tensors(
        self,
        input_tensors: List[Tensor],
    ) -> List[Tensor]:
        if not isinstance(input_tensors, list):
            ValueError("Input Tensors should be passed as a list of Tensors")

        validated_input_tensors = []

        for input_tensor in input_tensors:
            if not isinstance(input_tensor, Tensor):
                ValueError(
                    f"Expected all inputs to be of type Tensor. \
                    Got {type(input_tensor)}"
                )
            else:
                validated_input_tensors.append(input_tensor)

        return validated_input_tensors

    def input_broadcast_shape(self, input_tensors: List[Tensor]) -> Tensor:
        for tensor in input_tensors:
            if not tensor.should_broadcast:
                return None
        try:
            return np.broadcast_shapes(
                *(tensor.shape for tensor in input_tensors)
            )
        except ValueError:
            return None

    def forward(self, input_tensors: List[Tensor]) -> Tensor:
        input_tensors = self.validate_input_tensors(input_tensors)
        self.requires_grad(input_tensors)
        self._forward(input_tensors)

    @abstractmethod
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        raise NotImplementedError(
            f"Forward method not implemented for operator {self}",
        )

    @abstractmethod
    def backward(self, input_tensors: List[Tensor]) -> None:
        raise NotImplementedError(
            f"Backward pass not implemented for operator {self}",
        )


class Add(Operation):

    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 2:
            ValueError(f"Expect 2 input tensors but got {len(input_tensors)}")

        a, b = input_tensors

        result = Tensor(a.data + b.data, self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a, b = input_tensors
        a.grad_fn = lambda output_grad: output_grad * 1.0
        b.grad_fn = lambda output_grad: output_grad * 1.0


def add(input_tensors: List[Tensor]) -> Tensor:
    operation = Add()
    return operation(input_tensors)