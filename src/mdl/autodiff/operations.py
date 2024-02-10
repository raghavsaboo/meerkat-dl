from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from mdl.autodiff.dcgraph import DCGraph
from mdl.tensor import Parameter
from mdl.tensor import Tensor


class Operation(ABC):

    global_dc_graph = DCGraph()

    def __call__(
        self,
        input_tensors: List[Union[Tensor, Parameter]],
        *args: Any,
        **kwargs: Any,
    ) -> Union[Tensor, Parameter]:
        return self.forward(input_tensors, *args, **kwargs)

    def set_requires_grad(self, input_tensors: List[Tensor]) -> None:
        self.requires_grad = any(
            [tensor.requires_grad for tensor in input_tensors],
        )

    def validate_input_tensors(
        self,
        input_tensors: List[Union[Tensor, Parameter]],
    ) -> List[Union[Tensor, Parameter]]:
        if not isinstance(input_tensors, list):
            ValueError(
                "Input should be passed as a list of Tensors/Parameters"
            )

        validated_input_tensors = []

        for input_tensor in input_tensors:
            if not isinstance(input_tensor, (Tensor, Parameter)):
                ValueError(
                    f"Expected all inputs to be of type Tensor or Parameter. \
                    Got {type(input_tensor)}"
                )
            else:
                validated_input_tensors.append(input_tensor)

        return validated_input_tensors

    def input_broadcast_shape(
        self, input_tensors: List[Union[Tensor, Parameter]]
    ) -> Union[None, Tuple[int, ...]]:
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
        input_tensors: List[Union[Tensor, Parameter]],
        *args: Any,
        **kwargs: Any,
    ) -> Union[Tensor, Parameter]:
        input_tensors = self.validate_input_tensors(input_tensors)
        self.set_requires_grad(input_tensors)
        result = self._forward(input_tensors, *args, **kwargs)
        return result

    @abstractmethod
    def _forward(
        self,
        input_tensors: List[Union[Tensor, Parameter]],
        *args: Any,
        **kwargs: Any,
    ) -> Union[Tensor, Parameter]:
        raise NotImplementedError(
            f"Forward method not implemented for operator {self}",
        )

    @abstractmethod
    def backward(self, input_tensors: List[Union[Tensor, Parameter]]) -> None:
        raise NotImplementedError(
            f"Backward pass not implemented for operator {self}",
        )


class ParameterOperation(Operation, ABC):

    def __init__(self):
        super().__init__()
        self._eval = False

    def aggregate_parameters(
        self, as_list: bool = False
    ) -> List[Parameter] | Dict[str, Parameter]:
        parameters: Dict[str, Parameter] = dict()
        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                parameters[name] = value

        return list(parameters.values()) if as_list else parameters

    @property
    def eval(self):
        return self._eval

    @eval.setter
    def eval(self, value: bool = False):
        parameters = self.aggregate_parameters(
            as_list=True
        )  # type: ignore[union-attr]
        for param in parameters:
            param.eval = value  # type: ignore[union-attr]

        self._eval = value

    @abstractmethod
    def _forward(
        self,
        input_tensors: List[Union[Tensor, Parameter]],
        *args: Any,
        **kwargs: Any,
    ) -> Union[Tensor, Parameter]:
        raise NotImplementedError(
            f"Forward method not implemented for operator {self}",
        )

    @abstractmethod
    def backward(self, input_tensors: List[Union[Tensor, Parameter]]) -> None:
        raise NotImplementedError(
            f"Backward pass not implemented for operator {self}",
        )


# TODO: add conv2d, conv3d, dropout, rnn cell,
# TODO: lstm cell as parameter operations


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
    def _forward(
        self, input_tensors: List[Tensor], axis: Union[int, None]
    ) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]

        self.axis = axis

        result = Tensor(
            np.sum(a.data, axis=self.axis),
            self.requires_grad,
        )

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        tensor = input_tensors[0]
        if self.axis is None:
            tensor.grad_fn = (
                lambda output_grad: np.ones_like(tensor.data) * output_grad
            )
        else:
            tensor.grad_fn = lambda output_grad: np.expand_dims(
                output_grad, axis=self.axis
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


def sum_tensors(
    input_tensors: List[Tensor], axis: Union[int, None] = None
) -> Tensor:
    operation = Sum()
    return operation(input_tensors, axis)


def flatten(input_tensors: List[Tensor]) -> Tensor:
    operation = Flatten()
    return operation(input_tensors)


def transpose(input_tensors: List[Tensor]) -> Tensor:
    operation = Transpose()
    return operation(input_tensors)


def reshape(input_tensors: List[Tensor], new_shape: Tuple[int]) -> Tensor:
    operation = Reshape()
    return operation(input_tensors, new_shape)


class Concatenate(Operation):
    def _forward(self, input_tensors: List[Tensor], axis: int = 0) -> Tensor:
        result = Tensor(
            np.concatenate([t.data for t in input_tensors], axis=axis),
            self.requires_grad,
        )

        self.global_dc_graph.add_edge(result, input_tensors)

        self.axis = axis

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        indices = np.cumsum(
            [t.data.shape[self.axis] for t in input_tensors[:-1]]
        )
        a = input_tensors[-1]

        def grad_fn(output_grad):
            return np.split(output_grad, indices, axis=self.axis)[:-1]

        a.grad_fn = grad_fn


def concatenate(input_tensors: List[Tensor], axis: int = 0) -> Tensor:
    operation = Concatenate()
    return operation(input_tensors, axis)


# class Max(Operation):
#     def _forward(self, input_tensors: List[Tensor], axis: int = 0) -> Tensor:
#         if len(input_tensors) != 1:
#             raise ValueError(
#                 f"Expect 1 input tensor but got {len(input_tensors)}"
#             )

#         a = input_tensors[0]
#         self.axis = axis

#         result = Tensor(np.max(a.data, axis=self.axis), self.requires_grad)

#         self.global_dc_graph.add_edge(result, input_tensors)

#         result.backward_fn = self.backward
#         result.parent_broadcast_shape = self.input_broadcast_shape(
#             input_tensors
#         )

#         return result

#     def backward(self, input_tensors: List[Tensor]) -> None:
#         a = input_tensors[0]
#         a.grad_fn = lambda output_grad: output_grad * (
#             a.data == np.max(a.data, axis=self.axis)
#         )


# def max_operation(input_tensors: List[Tensor], axis: int = 0) -> Tensor:
#     operation = Max()
#     return operation(input_tensors, axis)


class Min(Operation):
    def _forward(self, input_tensors: List[Tensor], axis: int = 0) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        self.axis = axis

        result = Tensor(np.min(a.data, axis=self.axis), self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad * (
            a.data == np.min(a.data, axis=self.axis)
        )


def min_operation(input_tensors: List[Tensor], axis: int = 0) -> Tensor:
    operation = Min()
    return operation(input_tensors, axis)


class Mean(Operation):
    def _forward(self, input_tensors: List[Tensor], axis: int = 0) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        self.axis = axis

        result = Tensor(np.mean(a.data, axis=axis), self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad * (1 / a.data.size)


def mean(input_tensors: List[Tensor], axis: int = 0) -> Tensor:
    operation = Mean()
    return operation(input_tensors, axis)
