from __future__ import annotations

import numpy as np
from mdl.autodiff.utils import gradient_checker
from mdl.net.linear import LinearLayer
from mdl.net.loss import MeanSquaredErrorLoss
from mdl.net.sequence import RNNLayer
from mdl.tensor import Tensor


def test_linear():
    input_size, output_size = 20, 10
    batch_size = 10

    # Create an instance of the Linear operation
    linear = LinearLayer(input_size, output_size)

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


def test_rnn_layer():
    input_size, hidden_size = 10, 20
    seq_length, batch_size = 5, 10

    # Create an instance of the RNNLayer
    rnn_layer = RNNLayer(input_size, hidden_size)

    # Generate random input sequence
    input_sequence = Tensor(np.random.rand(seq_length, batch_size, input_size))
    # Generate random target output for loss calculation
    target_output = Tensor(
        np.random.rand(seq_length, batch_size, hidden_size),
        requires_grad=False,
    )
    # target_output = Tensor(np.random.rand(batch_size, hidden_size))
    # Instantiate the loss function
    loss_fn = MeanSquaredErrorLoss()

    diff = gradient_checker(
        component=rnn_layer,
        input_tensor=input_sequence,
        target=target_output,
        loss_fn=loss_fn,
        epsilon=1e-7,
    )
    assert diff < 1e-7
