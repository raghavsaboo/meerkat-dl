from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from mdl.autodiff.dcgraph import DCGraph
from mdl.tensor import Tensor


class Operation(ABC):

    global_dc_graph = DCGraph()

    @property
    def requires_grad(self):
        pass

    @requires_grad.setter
    def requires_grad(self, input_tensors: list[Tensor]) -> None:
        self.requires_grad = any(
            [tensor.requires_grad for tensor in input_tensors],
        )

    def validate_input_tensors(
        self,
        input_tensors: list[Tensor],
    ) -> list[Tensor]:
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

    def forward(self, input_tensors: list[Tensor]) -> Tensor:
        input_tensors = self.validate_input_tensors(input_tensors)
        self.requires_grad(input_tensors)
        self._forward(input_tensors)

    @abstractmethod
    def _forward(self, input_tensors: list[Tensor]) -> Tensor:
        raise NotImplementedError(
            f"Forward method not implemented for operator {self}",
        )

    @abstractmethod
    def backward(self):
        raise NotImplementedError(
            f"Backward pass not implemented for operator {self}",
        )


class Add(Operation):

    def _forward(self, input_tensors: list[Tensor]) -> Tensor:
        if len(input_tensors) != 2:
            ValueError(f"Expect 2 input tensors but got {len(input_tensors)}")

        a = input_tensors[0]
        b = input_tensors[1]

        result = Tensor(a.data + b.data, self.requires_grad)

        self.global_dc_graph.add_tensor_node(a)

        return result

    def backward(self) -> Tensor:
        pass
