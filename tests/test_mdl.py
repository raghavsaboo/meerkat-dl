from __future__ import annotations

from mdl.autodiff.tensor import Tensor
from mdl.base.operations import Add


def test_tensor_addition():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=False)
    b = Tensor([1.0, 2.0, 3.0], requires_grad=False)

    addition_op = Add()

    assert addition_op.forward(a, b) == Tensor(
        [2.0, 4.0, 6.0],
        requires_grad=False,
    )
