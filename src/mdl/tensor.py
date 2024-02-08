from __future__ import annotations

from typing import Callable
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from mdl.autodiff.dcgraph import DCGraph
from mdl.utilities import unbroadcast

TensorDataTypes = Union[float, int, list, np.ndarray]


class Tensor:

    global_dc_graph = DCGraph()

    def __init__(
        self,
        data: TensorDataTypes,
        requires_grad: bool = False,
        should_broadcast: bool = True,
    ) -> None:
        self._data = self._convert_to_ndarray(data)
        self._requires_grad = requires_grad

        if self._requires_grad:
            self.set_gradients_to_zero()

        self._child_tensors: List[Tensor] = []
        self._parent_tensors: List[Tensor] = []
        self._parent_broadcast_shape: Tuple[int] | None = None
        self._grad_fn: Callable | None = None
        self._should_broadcast = should_broadcast
        self._backward_fn: Callable | None = None

    def __str__(self):
        return f"Tensor({self.data})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: TensorDataTypes):
        self._data = self._convert_to_ndarray(data)
        # changing the data means that the current gradient
        # is invalid
        self.set_gradients_to_zero()

    @property
    def should_broadcast(self):
        return self._should_broadcast

    @should_broadcast.setter
    def should_broadcast(self, value: bool):
        self._should_broadcast = value

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._data.shape)

    @property
    def grad(self) -> np.ndarray:
        return self._grad

    def accumulate_grad(self, gradient: TensorDataTypes):
        self._grad += gradient

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool = False):
        self._requires_grad = requires_grad
        # resetting grads after changing requesting grads
        self.set_gradients_to_zero()

    @property
    def child_tensors(self):
        return self._child_tensors

    def add_child_tensor(self, tensor: Tensor) -> None:
        assert isinstance(tensor, Tensor), "Expected Tensor type"
        self._child_tensors.append(tensor)

    @property
    def parent_tensors(self):
        return self._parent_tensors

    def add_parent_tensor(self, tensor: Tensor) -> None:
        assert isinstance(tensor, Tensor), "Expected Tensor type"
        self._parent_tensors.append(tensor)

    @property
    def parent_broadcast_shape(self):
        self._parent_broadcast_shape

    @parent_broadcast_shape.setter
    def parent_broadcast_shape(self, shape: tuple[int]):
        self._parent_broadcast_shape = shape

    @property
    def grad_fn(self):
        return self._grad_fn

    @grad_fn.setter
    def grad_fn(self, function: Callable):
        self._grad_fn = function if self.requires_grad else None

    @property
    def backward_fn(self):
        self._backward_fn

    @backward_fn.setter
    def backward_fn(self, function: Callable):
        self._backward_fn = function

    def set_gradients_to_zero(self):
        self._grad = np.zeros_like(self._data)

    @staticmethod
    def _convert_to_ndarray(data: TensorDataTypes) -> np.ndarray:
        assert isinstance(
            data,
            (float, int, list, np.ndarray),
        ), "Incompatible type for `data`. Expect float, int or numpy array."

        return np.array(data)

    def to_list(self):
        return self._data.tolist()

    def to_array(self):
        return self._data

    def __add__(self, b: Tensor) -> Tensor:
        from mdl.operations import add

        return add([self, b])

    def __radd__(self, b: Tensor) -> Tensor:
        from mdl.operations import add

        return add([b, self])

    def __iadd__(self, b: Tensor) -> Tensor:
        from mdl.operations import add

        return add([self, b])

    def __sub__(self, other: Tensor) -> Tensor:
        from mdl.operations import sub

        return sub([self, other])

    def __rsub__(self, other: Tensor) -> Tensor:
        from mdl.operations import sub

        return sub([other, self])

    def __isub__(self, other: Tensor) -> Tensor:
        from mdl.operations import sub

        return sub([self, other])

    def __mul__(self, other: Tensor) -> Tensor:
        from mdl.operations import mul

        return mul([self, other])

    def __rmul__(self, other: Tensor) -> Tensor:
        from mdl.operations import mul

        return mul([other, self])

    def __imul__(self, other: Tensor) -> Tensor:
        from mdl.operations import mul

        return mul([self, other])

    def __truediv__(self, other: Tensor) -> Tensor:
        from mdl.operations import div

        return div([self, other])

    def __rtruediv__(self, other: Tensor) -> Tensor:
        from mdl.operations import div

        return div([other, self])

    def __itruediv__(self, other: Tensor) -> Tensor:
        from mdl.operations import div

        return div([self, other])

    def __matmul__(self, other: Tensor) -> Tensor:
        from mdl.operations import dot

        return dot([self, other])

    def __rmatmul__(self, other: Tensor) -> Tensor:
        from mdl.operations import dot

        return dot([other, self])

    def __imatmul__(self, other: Tensor) -> Tensor:
        from mdl.operations import dot

        return dot([self, other])

    def __exp__(self) -> Tensor:
        from mdl.operations import exp

        return exp([self])

    def __log__(self) -> Tensor:
        from mdl.operations import log

        return log([self])

    def sum(self) -> Tensor:
        from mdl.operations import sum_tensors

        return sum_tensors([self])

    def flatten(self) -> Tensor:
        from mdl.operations import flatten

        return flatten([self])

    def transpose(self) -> Tensor:
        from mdl.operations import transpose

        return transpose([self])

    def reshape(self, new_shape: Tuple[int]) -> Tensor:
        from mdl.operations import reshape

        return reshape([self], new_shape)

    def pow(self, exponent: float) -> Tensor:
        from mdl.operations import power

        return power([self], exponent)

    def mean(self, axis: Union[int, None] = None) -> Tensor:
        from mdl.operations import mean

        return mean([self], axis)

    def min(self, axis: Union[int, None] = None) -> Tensor:
        from mdl.operations import min_operation

        return min_operation([self], axis)

    def max(self, axis: Union[int, None] = None) -> Tensor:
        from mdl.operations import max_operation

        return max_operation([self], axis)

    def concatenate(self, other: Tensor, axis: int = 0) -> Tensor:
        from mdl.operations import concatenate

        return concatenate([self, other], axis)

    def backward(self, output_grad: TensorDataTypes = 1.0):
        if not self.requires_grad:
            raise Exception(
                "Can call backward only from tensors with requires_grad = True"
            )

        output_grad = self._convert_to_ndarray(output_grad)

        if self.shape != output_grad.shape:
            raise Exception("Shapes of gradient and Tensor need to match.")

        self.accumulate_grad(output_grad)
        self.global_dc_graph.backpropogate(self)

    def backprop_calculation(self):
        for child in self.child_tensors:
            if self.requires_grad:
                child.backward_fn(*[tensor for tensor in child.parent_tensors])
                output_grad = child.grad
                local_grad = self.grad_fn(output_grad)
                local_grad = unbroadcast(
                    local_grad, self.shape, child.parent_broadcast_shape
                )
                local_grad = local_grad.reshape(self.shape)
                self.accumulate_grad(local_grad)


class Parameter(Tensor):

    def __init__(
        self,
        data: TensorDataTypes,
        requires_grad: bool = False,
    ) -> None:
        super().__init__(data, requires_grad)

    @property
    def frozen(self):
        return not self.requires_grad
