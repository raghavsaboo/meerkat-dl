from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Tuple

import numpy as np
from mdl.autodiff.dcgraph import DCGraph
from mdl.tensor import Tensor


class Operation(ABC):

    global_dc_graph = DCGraph()

    def __call__(
        self,
        input_tensors: List[Tensor],
        *args: Any,
        **kwargs: Any,
    ):
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

    def forward(
        self,
        input_tensors: List[Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        input_tensors = self.validate_input_tensors(input_tensors)
        self.requires_grad(input_tensors)
        self._forward(input_tensors, *args, **kwargs)

    @abstractmethod
    def _forward(
        self,
        input_tensors: List[Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
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


class Sub(Operation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 2:
            raise ValueError(
                f"Expect 2 input tensors but got {len(input_tensors)}"
            )

        a, b = input_tensors
        result = Tensor(a.data - b.data, self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a, b = input_tensors
        a.grad_fn = lambda output_grad: output_grad * 1.0
        b.grad_fn = lambda output_grad: output_grad * -1.0


class Mul(Operation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 2:
            raise ValueError(
                f"Expect 2 input tensors but got {len(input_tensors)}"
            )

        a, b = input_tensors
        result = Tensor(a.data * b.data, self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a, b = input_tensors
        a.grad_fn = lambda output_grad: output_grad * b.data
        b.grad_fn = lambda output_grad: output_grad * a.data


class Div(Operation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 2:
            raise ValueError(
                f"Expect 2 input tensors but got {len(input_tensors)}"
            )

        a, b = input_tensors
        result = Tensor(a.data / b.data, self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a, b = input_tensors
        a.grad_fn = lambda output_grad: output_grad / b.data
        b.grad_fn = lambda output_grad: output_grad * (-a.data / (b.data**2))


class Dot(Operation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 2:
            raise ValueError(
                f"Expect 2 input tensors but got {len(input_tensors)}"
            )

        a, b = input_tensors
        result = Tensor(np.dot(a.data, b.data), self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a, b = input_tensors
        a.grad_fn = lambda output_grad: np.dot(output_grad, b.data.T)
        b.grad_fn = lambda output_grad: np.dot(a.data.T, output_grad)


class Exp(Operation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(np.exp(a.data), self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad * np.exp(a.data)


class Log(Operation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(np.log(a.data), self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad / a.data


class Sum(Operation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        result = Tensor(
            np.sum([tensor.data for tensor in input_tensors]),
            self.requires_grad,
        )

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        for tensor in input_tensors:
            tensor.grad_fn = lambda output_grad: output_grad * np.ones_like(
                tensor.data
            )


class Flatten(Operation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(a.data.flatten(), self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad.reshape(a.shape)


class Transpose(Operation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(a.data.T, self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad.T


class Reshape(Operation):
    def _forward(
        self, input_tensors: List[Tensor], new_shape: Tuple[int]
    ) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(a.data.reshape(new_shape), self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad.reshape(a.shape)


class Pow(Operation):
    def _forward(self, input_tensors: List[Tensor], exponent: float) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        self.exponent = exponent

        result = Tensor(a.data**self.exponent, self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad * (
            a.data ** (self.exponent - 1)
        )


def power(input_tensors: List[Tensor], exponent: float) -> Tensor:
    operation = Pow()
    return operation(input_tensors, exponent)


def add(input_tensors: List[Tensor]) -> Tensor:
    operation = Add()
    return operation(input_tensors)


def sub(input_tensors: List[Tensor]) -> Tensor:
    operation = Sub()
    return operation(input_tensors)


def mul(input_tensors: List[Tensor]) -> Tensor:
    operation = Mul()
    return operation(input_tensors)


def div(input_tensors: List[Tensor]) -> Tensor:
    operation = Div()
    return operation(input_tensors)


def dot(input_tensors: List[Tensor]) -> Tensor:
    operation = Dot()
    return operation(input_tensors)


def exp(input_tensors: List[Tensor]) -> Tensor:
    operation = Exp()
    return operation(input_tensors)


def log(input_tensors: List[Tensor]) -> Tensor:
    operation = Log()
    return operation(input_tensors)


def sum_tensors(input_tensors: List[Tensor]) -> Tensor:
    operation = Sum()
    return operation(input_tensors)


def flatten(input_tensors: List[Tensor]) -> Tensor:
    operation = Flatten()
    return operation(input_tensors)


def transpose(input_tensors: List[Tensor]) -> Tensor:
    operation = Transpose()
    return operation(input_tensors)


def reshape(input_tensors: List[Tensor], new_shape: Tuple[int]) -> Tensor:
    operation = Reshape()
    return operation(input_tensors, new_shape)
