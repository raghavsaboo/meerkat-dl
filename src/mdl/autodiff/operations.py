from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from mdl.autodiff.dcgraph import DCGraph
from mdl.tensor import Parameter
from mdl.tensor import Tensor
from typing_extensions import override


class Operation(ABC):
    """
    The Operation class is an abstract base class (ABC) that
    defines the interface for operations in the computational graph.

    It includes methods for forward computation (_forward),
    backward computation (backward), and other utility methods.

    Operations are meant to be used for basic mathematical
    operations like addition, multiplication, etc.

    Instances of Operation are not meant to have any
    learnable parameters (like weights or biases).
    """

    global_dc_graph = DCGraph()

    def __call__(
        self,
        input_tensors: List[Tensor | Parameter],
        *args: Any,
        **kwargs: Any,
    ) -> Tensor | Parameter:
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

    @staticmethod
    def input_broadcast_shape(
        input_tensors: List[Union[Tensor, Parameter]]
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
    """
    The ParameterOperation class is a subclass of Operation
    and extends it to handle operations with learnable parameters.

    It introduces the concept of parameters, which are
    tensors with associated gradients that can be updated during training.

    It includes additional methods for managing parameters,
    such as aggregate_parameters and eval.

    Subclasses of ParameterOperation are expected to define
    the forward and backward computations specific to their operation.
    """

    def __init__(self):
        super().__init__()
        self._eval = False

    def aggregate_parameters_as_dict(self) -> Dict[str, Parameter]:
        parameters: Dict[str, Parameter] = dict()
        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                parameters[name] = value

        return parameters

    def aggregate_parameters_as_list(self) -> List[Parameter]:
        return list(self.aggregate_parameters_as_dict().values())

    @override
    def set_requires_grad(self, input_tensors: List[Tensor]) -> None:
        self.requires_grad = any(
            [tensor.requires_grad for tensor in input_tensors]
            + self.aggregate_parameters_as_list(),
        )

    @property
    def eval(self):
        return self._eval

    @eval.setter
    def eval(self, value: bool = False):
        parameters = self.aggregate_parameters_as_list()
        for param in parameters:
            param.eval = value

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


class Add(Operation):

    def __init__(self):
        super().__init__()

    def _forward(self, input_tensors: List[Tensor | Parameter]) -> Tensor:
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

    def backward(self, input_tensors: List[Tensor | Parameter]) -> None:
        a, b = input_tensors
        a.grad_fn = lambda output_grad: output_grad * 1.0
        b.grad_fn = lambda output_grad: output_grad * 1.0


class Sub(Operation):

    def __init__(self):
        super().__init__()

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


class BatchedMatMul(Operation):
    def _forward(
        self,
        input_tensors: List[Union[Tensor, Parameter]],
    ) -> Tensor:
        if len(input_tensors) != 2:
            raise ValueError(
                "BatchedMatMul operation expects 2 input tensors."
            )

        batch_matrix1, batch_matrix2 = input_tensors

        # Check the shapes of input tensors
        if len(batch_matrix1.shape) != 3 or len(batch_matrix2.shape) != 3:
            raise ValueError("Input tensors must be 3D batched matrices.")

        batch_size, m, n = batch_matrix1.shape
        _, _, p = batch_matrix2.shape

        # Reshape matrices to 2D for batched multiplication
        flat_matrix1 = batch_matrix1.data.reshape(batch_size * m, n)
        flat_matrix2 = batch_matrix2.data.reshape(batch_size * n, p)

        # Perform batched matrix multiplication
        flat_result = np.dot(flat_matrix1, flat_matrix2)

        # Reshape the result back to 3D
        result_data = flat_result.reshape(batch_size, m, p)
        result = Tensor(result_data, requires_grad=self.requires_grad)

        # Add edges between output tensor and input tensors
        self.global_dc_graph.add_edge(result, [batch_matrix1, batch_matrix2])

        return result

    def backward(self, input_tensors: List[Union[Tensor, Parameter]]) -> None:
        if len(input_tensors) != 2:
            raise ValueError(
                "BatchedMatMul operation backward expects 2 input tensors."
            )

        batch_matrix1, batch_matrix2 = input_tensors

        # Compute gradients with respect to input tensors
        batch_matrix1.grad_fn = lambda output_grad: np.dot(
            output_grad, batch_matrix2.data.transpose(0, 2, 1)
        )
        batch_matrix2.grad_fn = lambda output_grad: np.dot(
            batch_matrix1.data.transpose(0, 2, 1), output_grad
        )


class Mul(Operation):
    """
    Hadamard product (elementwise multiplication)
    """

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
        b.grad_fn = lambda output_grad: output_grad * (
            -1 * a.data / np.power(b.data, 2)
        )


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

        result = Tensor(np.power(a.data, self.exponent), self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = (
            lambda output_grad: self.exponent
            * output_grad
            * (a.data ** (self.exponent - 1))
        )


def bmm(input_tensors: List[Tensor]) -> Tensor:
    operation = BatchedMatMul()
    return operation(input_tensors)


def power(input_tensors: List[Tensor], exponent: float) -> Tensor:
    operation = Pow()
    return operation(input_tensors, exponent)


def add(input_tensors: List[Tensor | Parameter]) -> Tensor | Parameter:
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


class Max(Operation):
    def _forward(
        self, input_tensors: List[Tensor], axis: int | None = None
    ) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        self.axis = axis

        result = Tensor(
            np.max(a.data, axis=self.axis, keepdims=True), self.requires_grad
        )

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        max_values = np.max(a.data, axis=self.axis, keepdims=True)
        mask = (a.data == max_values).astype(float)
        a.grad_fn = lambda output_grad: output_grad * mask


def max_operation(
    input_tensors: List[Tensor], axis: int | None = None
) -> Tensor:
    operation = Max()
    return operation(input_tensors, axis)


class Min(Operation):
    def _forward(
        self, input_tensors: List[Tensor], axis: int | None = None
    ) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        self.axis = axis

        result = Tensor(
            np.min(a.data, axis=self.axis, keepdims=True), self.requires_grad
        )

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        min_values = np.min(a.data, axis=self.axis, keepdims=True)
        mask = (a.data == min_values).astype(float)
        a.grad_fn = lambda output_grad: output_grad * mask


def min_operation(
    input_tensors: List[Tensor], axis: int | None = None
) -> Tensor:
    operation = Min()
    return operation(input_tensors, axis)


class Mean(Operation):
    def _forward(
        self, input_tensors: List[Tensor], axis: int | None = None
    ) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        self.axis = axis

        result = Tensor(
            np.mean(a.data, axis=self.axis, keepdims=True), self.requires_grad
        )

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        if self.axis is None:
            a.grad_fn = (
                lambda output_grad: output_grad
                * np.ones_like(a.data)
                / a.data.size
            )
        else:
            shape = list(a.shape)
            shape[self.axis] = 1
            a.grad_fn = (
                lambda output_grad: output_grad
                * np.ones(shape)
                / a.data.shape[self.axis]
            )


def mean(input_tensors: List[Tensor], axis: int | None = None) -> Tensor:
    operation = Mean()
    return operation(input_tensors, axis)


class Abs(Operation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(np.abs(a.data), self.requires_grad)
        self.global_dc_graph.add_edge(result, input_tensors)
        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )
        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad * np.sign(a.data)


def abs_operation(input_tensors: List[Tensor]) -> Tensor:
    operation = Abs()
    return operation(input_tensors)


class Sqrt(Operation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(np.sqrt(a.data), self.requires_grad)
        self.global_dc_graph.add_edge(result, input_tensors)
        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )
        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad / (2 * np.sqrt(a.data))


class Sin(Operation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(np.sin(a.data), self.requires_grad)
        self.global_dc_graph.add_edge(result, input_tensors)
        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )
        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad * np.cos(a.data)


class Cos(Operation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(np.cos(a.data), self.requires_grad)
        self.global_dc_graph.add_edge(result, input_tensors)
        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )
        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: -output_grad * np.sin(a.data)


def sqrt(input_tensors: List[Tensor]) -> Tensor:
    operation = Sqrt()
    return operation(input_tensors)


def sin(input_tensors: List[Tensor]) -> Tensor:
    operation = Sin()
    return operation(input_tensors)


def cos(input_tensors: List[Tensor]) -> Tensor:
    operation = Cos()
    return operation(input_tensors)


class Slice(Operation):
    def __init__(self, indices: tuple):
        super().__init__()
        self.indices = indices

    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError("Slice operation expects 1 input tensor.")

        input_tensor = input_tensors[0]
        sliced_data = input_tensor.data[self.indices]
        result = Tensor(sliced_data, requires_grad=input_tensor.requires_grad)

        # Register this operation in the global computation graph
        self.global_dc_graph.add_edge(result, input_tensors)

        # Set backward function and parent broadcast shape
        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor | Parameter]) -> None:
        parent_tensor = input_tensors[0]

        parent_tensor.grad_fn = lambda output_grad: np.add.at(
            parent_tensor.grad, self.indices, output_grad
        )


def slicer(input_tensors: List[Tensor], indices) -> Tensor:
    operation = Slice(indices)
    return operation(input_tensors)


class Stack(Operation):
    def __init__(self, axis: int = 0):
        super().__init__()
        self.axis = axis

    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) < 1:
            raise ValueError(
                "Expect at least 1 input tensor for Stack operation"
            )

        stacked_data = np.stack(
            [tensor.data for tensor in input_tensors], axis=self.axis
        )
        result = Tensor(stacked_data, self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)
        # Set backward function and parent broadcast shape
        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        num_tensors = len(input_tensors)

        for i, tensor in enumerate(input_tensors):
            # Define a grad_fn for each tensor to correctly
            # slice the output_grad
            # Capture the current loop variables in the lambda's
            # default arguments to ensure they persist
            def stack_grad_fn(output_grad, i, axis, num_tensors):
                return self._slice_grad(output_grad, i, axis, num_tensors)

            tensor.grad_fn = partial(
                stack_grad_fn, i=i, axis=self.axis, num_tensors=num_tensors
            )

    @staticmethod
    def _slice_grad(output_grad, index, axis, num_tensors):
        # Compute the total gradient size along the
        # stacking axis and the slice size for each tensor
        total_grad_size = output_grad.shape[axis]
        slice_size = total_grad_size // num_tensors
        start_idx = index * slice_size
        end_idx = start_idx + slice_size

        # Extract the slice for the current tensor
        # directly, without additional reshaping
        grad_slice = np.take(output_grad, range(start_idx, end_idx), axis=axis)

        return grad_slice


def stack(input_tensors: List[Tensor], axis: int = 0) -> Tensor:
    operation = Stack(axis)
    return operation(input_tensors)
