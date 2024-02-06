from __future__ import annotations

from typing import Callable
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from mdl.autodiff.dcgraph import DCGraph
from mdl.base.operations import Add

TensorDataTypes = Union[float, int, list, np.ndarray]


class Tensor:

    global_dc_graph = DCGraph()

    def __init__(
        self,
        data: TensorDataTypes,
        requires_grad: bool = False,
    ) -> None:
        self._data = self._convert_to_ndarray(data)
        self._requires_grad = requires_grad

        if self._requires_grad:
            self.set_gradients_to_zero()

        self._child_tensors: List[Tensor] = []
        self._parent_tensors: List[Tensor] = []
        self._parent_broadcast_shape: Tuple[int] | None = None
        self._grad_fn: Callable | None = None

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
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

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
        self._grad_fn

    @grad_fn.setter
    def grad_fn(self, function: Callable):
        self._grad_fn = function

    def set_gradients_to_zero(self):
        self._grad = np.zeros_like(self._data)

    @staticmethod
    def _convert_to_ndarray(data: TensorDataTypes) -> np.ndarray:
        assert not isinstance(
            data,
            (float, int, list, np.ndarray),
        ), "Incompatible type for `data`. Expect float, int or numpy array."

        return np.array(data)

    def to_list(self):
        return self._data.tolist()

    def to_array(self):
        return self._data

    def __add__(self, b: TensorDataTypes) -> Tensor:
        addition_op = Add()
        return addition_op.forward(self, b)


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
