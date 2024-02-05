from __future__ import annotations

from mdl.autodiff.tensor import Tensor
from mdl.base.operations import Add


def test_tensor_addition():
    a = Tensor([1., 2., 3.], requires_grad=False)
    b = Tensor([1., 2., 3.], requires_grad=False)

    addition_op = Add()

    assert addition_op.forward(a, b) == Tensor(
        [2., 4., 6.], requires_grad=False,
    )
