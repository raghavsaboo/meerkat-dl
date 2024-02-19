import numpy as np

from mdl.net.loss import MeanSquaredErrorLoss
from mdl.autodiff.linear import Linear
from mdl.autodiff.utils import gradient_checker
from mdl.tensor import Tensor

def test_linear():
    input_size, output_size = 20, 10
    batch_size = 10

    # Create an instance of the Linear operation
    linear = Linear(input_size, output_size)

    # Generate random input tensor
    input_tensor = Tensor(np.random.rand(batch_size, input_size))
    # Generate a random target tensor for loss calculation
    target_tensor = Tensor(np.random.rand(batch_size, output_size))
    # Instantiate the real loss function from your framework
    loss_fn = MeanSquaredErrorLoss()  # Using the correct class name
    diff = gradient_checker(
        component=linear,
        input_tensor=input_tensor,
        target=target_tensor,
        loss_fn=loss_fn,
        epsilon=1e-7,
    )
    assert diff < 1e-7
