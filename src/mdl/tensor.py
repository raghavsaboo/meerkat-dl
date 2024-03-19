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
    """
    Represents a tensor with autograd capabilities.
    """

    __slots__ = [
        "_data",
        "_requires_grad",
        "_child_tensors",
        "_parent_tensors",
        "_parent_broadcast_shape",
        "_grad_fn",
        "_should_broadcast",
        "_backward_fn",
        "_grad",
        "_eval",
    ]

    global_dc_graph = DCGraph()
    
    def __init__(
        self,
        data: TensorDataTypes,
        requires_grad: bool = False,
        should_broadcast: bool = True,
    ) -> None:
        """
        Initializes a Tensor object.

        Args:
            data (TensorDataTypes): The data of the tensor.
            requires_grad (bool): Whether the tensor requires gradient computation.
            should_broadcast (bool): Whether broadcasting should be applied in operations.
        """
        self._data = self._convert_to_ndarray(data)
        self._requires_grad = requires_grad

        if self._requires_grad:
            self.zero_grad()

        self._child_tensors: List[Tensor] = []
        self._parent_tensors: List[Tensor] = []
        self._parent_broadcast_shape: Tuple[int] | None = None
        self._grad_fn: Callable | None = None
        self._should_broadcast = should_broadcast
        self._backward_fn: Callable | None = None
        self._eval = False

    def __str__(self):
        return f"Tensor({self.data})"

    def __repr__(self):
        return f"Tensor({self.data})"
    
    def __hash__(self):
        return id(self)
    
    def __getitem__(self, key):
        from mdl.autodiff.operations import slicer
        if isinstance(key, (slice, int, tuple)):
            return slicer([self], key)
        else:
            raise TypeError("Unsupported key type for Tensor slicing.")
         

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: TensorDataTypes):
        self._data = self._convert_to_ndarray(data)

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
        if self.shape != gradient.shape:
            raise ValueError("Shapes of gradient and Tensor need to match")
        self._grad += gradient

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool = False):
        self._requires_grad = requires_grad
        # resetting grads after changing requesting grads
        self.zero_grad()

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
        return self._parent_broadcast_shape

    @parent_broadcast_shape.setter
    def parent_broadcast_shape(self, shape: Tuple[int]):
        self._parent_broadcast_shape = shape

    @property
    def grad_fn(self):
        return self._grad_fn

    @grad_fn.setter
    def grad_fn(self, function: Callable):
        self._grad_fn = function if self.requires_grad else None

    @property
    def backward_fn(self):
        return self._backward_fn

    @backward_fn.setter
    def backward_fn(self, function: Callable):
        self._backward_fn = function

    def zero_grad(self):
        self._grad = np.zeros_like(self._data, dtype=float)

    @staticmethod
    def _convert_to_ndarray(data: TensorDataTypes) -> np.ndarray:
        assert isinstance(
            data,
            (float, int, list, np.ndarray, np.generic),
        ), "Incompatible type for `data`. Expect float, int or numpy array."

        return np.array(data, dtype=np.float64)

    @classmethod
    def zeros(cls, shape: Union[int, Tuple[int, ...]], requires_grad: bool = False) -> Tensor:
        """
        Creates a tensor filled with zeros.

        Args:
            shape (Union[int, Tuple[int, ...]]): The shape of the tensor.
            requires_grad (bool): Whether the tensor requires gradient computation.

        Returns:
            Tensor: A tensor filled with zeros.
        """
        return cls(np.zeros(shape), requires_grad=requires_grad)

    @classmethod
    def ones(cls, shape: Union[int, Tuple[int, ...]], requires_grad: bool = False) -> Tensor:
        """
        Creates a tensor filled with ones.

        Args:
            shape (Union[int, Tuple[int, ...]]): The shape of the tensor.
            requires_grad (bool): Whether the tensor requires gradient computation.

        Returns:
            Tensor: A tensor filled with ones.
        """
        return cls(np.ones(shape), requires_grad=requires_grad)

    @classmethod
    def random(cls, shape: Union[int, Tuple[int, ...]], requires_grad: bool = False) -> Tensor:
        """
        Creates a tensor filled with random values.

        Args:
            shape (Union[int, Tuple[int, ...]]): The shape of the tensor.
            requires_grad (bool): Whether the tensor requires gradient computation.

        Returns:
            Tensor: A tensor filled with random values.
        """
        return cls(np.random.rand(*shape), requires_grad=requires_grad)

    def to_list(self):
        return self._data.tolist()

    def to_array(self):
        return self._data

    def __add__(self, b: Tensor) -> Tensor:
        from mdl.autodiff.operations import add

        return add([self, b])

    def __radd__(self, b: Tensor) -> Tensor:
        from mdl.autodiff.operations import add

        return add([b, self])

    def __iadd__(self, b: Tensor) -> Tensor:
        from mdl.autodiff.operations import add

        return add([self, b])

    def __sub__(self, other: Tensor) -> Tensor:
        from mdl.autodiff.operations import sub

        return sub([self, other])

    def __rsub__(self, other: Tensor) -> Tensor:
        from mdl.autodiff.operations import sub

        return sub([other, self])

    def __isub__(self, other: Tensor) -> Tensor:
        from mdl.autodiff.operations import sub

        return sub([self, other])

    def __mul__(self, other: Tensor) -> Tensor:
        from mdl.autodiff.operations import mul

        return mul([self, other])

    def __rmul__(self, other: Tensor) -> Tensor:
        from mdl.autodiff.operations import mul

        return mul([other, self])

    def __imul__(self, other: Tensor) -> Tensor:
        from mdl.autodiff.operations import mul

        return mul([self, other])

    def __truediv__(self, other: Tensor) -> Tensor:
        from mdl.autodiff.operations import div

        return div([self, other])

    def __rtruediv__(self, other: Tensor) -> Tensor:
        from mdl.autodiff.operations import div

        return div([other, self])

    def __itruediv__(self, other: Tensor) -> Tensor:
        from mdl.autodiff.operations import div

        return div([self, other])

    def __matmul__(self, other: Tensor) -> Tensor:
        from mdl.autodiff.operations import dot

        return dot([self, other])

    def __rmatmul__(self, other: Tensor) -> Tensor:
        from mdl.autodiff.operations import dot

        return dot([other, self])

    def __imatmul__(self, other: Tensor) -> Tensor:
        from mdl.autodiff.operations import dot

        return dot([self, other])

    def __exp__(self) -> Tensor:
        from mdl.autodiff.operations import exp

        return exp([self])

    def __pow__(self, exponent: float) -> Tensor:
        from mdl.autodiff.operations import power

        return power([self], exponent)

    def __abs__(self) -> Tensor:
        from mdl.autodiff.operations import abs_operation

        return abs_operation([self])

    def __eq__(self, other: Tensor) -> Tensor:
        return Tensor(self._data == other._data)

    def __ne__(self, other: Tensor) -> Tensor:
        return Tensor(self._data != other._data)

    def __lt__(self, other: Tensor) -> Tensor:
        return Tensor(self._data < other._data)

    def __gt__(self, other: Tensor) -> Tensor:
        return Tensor(self._data > other._data)

    def __le__(self, other: Tensor) -> Tensor:
        return Tensor(self._data <= other._data)

    def __ge__(self, other: Tensor) -> Tensor:
        return Tensor(self._data >= other._data)

    def bmm(self, other: Tensor) -> Tensor:
        from mdl.autodiff.operations import bmm

        return bmm([self, other])

    def sum(self, axis: Union[int, None] = None) -> Tensor:
        from mdl.autodiff.operations import sum_tensors

        return sum_tensors([self], axis)

    def flatten(self) -> Tensor:
        from mdl.autodiff.operations import flatten

        return flatten([self])

    def transpose(self) -> Tensor:
        from mdl.autodiff.operations import transpose

        return transpose([self])

    def reshape(self, new_shape: Tuple[int]) -> Tensor:
        from mdl.autodiff.operations import reshape

        return reshape([self], new_shape)

    def log(self) -> Tensor:
        from mdl.autodiff.operations import log

        return log([self])

    def mean(self, axis: Union[int, None] = None) -> Tensor:
        from mdl.autodiff.operations import mean

        return mean([self], axis)

    def min(self, axis: Union[int, None] = None) -> Tensor:
        from mdl.autodiff.operations import min_operation

        return min_operation([self], axis)

    def max(self, axis: Union[int, None] = None) -> Tensor:
        from mdl.autodiff.operations import max_operation

        return max_operation([self], axis)

    def concatenate(self, other: Tensor, axis: int = 0) -> Tensor:
        from mdl.autodiff.operations import concatenate

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
        # TODO: add reset graph here if retain_graph=False
        # --> to free up buffers

    def backprop_calculation(self):
        for child in self.child_tensors:
            if self.requires_grad:
                parent_tensors = [
                    tensor
                    for tensor in child.parent_tensors
                ]
                child.backward_fn(parent_tensors)
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
        requires_grad: bool = True,
    ) -> None:
        super().__init__(data, requires_grad)
        self._eval = False

    @property
    def eval(self) -> bool:
        return self._eval

    @eval.setter
    def eval(self, value: bool = False) -> None:
        self._eval = value

    @property
    def frozen(self):
        return not self.requires_grad
