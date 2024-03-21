from __future__ import annotations

import numpy as np
from mdl.autodiff.operations import stack
from mdl.net.activation import Activation
from mdl.net.activation import ActivationFunction
from mdl.net.components import Layer
from mdl.net.components import Module
from mdl.tensor import Parameter
from mdl.tensor import Tensor


class RNNCell(Layer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        activation: str | Activation = "tanh",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

        # Initialize weights and biases
        self.W_ih = Parameter(np.random.randn(input_size, hidden_size) * 0.01)
        self.W_hh = Parameter(np.random.randn(hidden_size, hidden_size) * 0.01)
        if use_bias:
            self.b_ih: Parameter = Parameter(np.zeros((1, hidden_size)))
            self.b_hh: Parameter = Parameter(np.zeros((1, hidden_size)))

        # Activation function
        self.activation = ActivationFunction.get(activation)

    def forward(self, input_tensor: Tensor, hidden_state: Tensor) -> Tensor:
        # Using MatMul operation for matrix multiplication
        input_contrib = input_tensor @ self.W_ih
        hidden_contrib = hidden_state @ self.W_hh

        if self.use_bias:
            # Using Add operation for adding bias
            input_contrib += self.b_ih
            hidden_contrib += self.b_hh

        # Combining input and hidden contributions
        combined = input_contrib + hidden_contrib

        new_hidden_state = self.activation([combined])

        return new_hidden_state


class RNNLayer(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.rnn_cell = RNNCell(
            input_size, hidden_size
        )  # Assuming RNNCell also follows the framework's conventions
        self.hidden_size = hidden_size

    def forward(self, input_tensor: Tensor) -> Tensor:
        sequence_length, batch_size, _ = input_tensor.shape
        hidden_states = []

        # Initialize the hidden state as zeros
        hidden_state = Tensor(
            np.zeros((batch_size, self.hidden_size)), requires_grad=True
        )

        for t in range(sequence_length):
            current_input = input_tensor[
                t, :, :
            ]  # Slice to get the input for timestamp t
            hidden_state = self.rnn_cell.forward(current_input, hidden_state)
            hidden_states.append(hidden_state)

        hidden_states_tensor = stack(hidden_states, axis=0)

        return hidden_states_tensor


class GRUCell(Layer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation="tanh",
        recurrent_activation="sigmoid",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight matrices for the input
        self.W_xz = Parameter(
            np.random.randn(input_size, hidden_size) * 0.01
        )  # Update gate
        self.W_xr = Parameter(
            np.random.randn(input_size, hidden_size) * 0.01
        )  # Reset gate
        self.W_xh = Parameter(
            np.random.randn(input_size, hidden_size) * 0.01
        )  # New memory content

        # Weight matrices for the hidden state
        self.W_hz = Parameter(
            np.random.randn(hidden_size, hidden_size) * 0.01
        )  # Update gate
        self.W_hr = Parameter(
            np.random.randn(hidden_size, hidden_size) * 0.01
        )  # Reset gate
        self.W_hh = Parameter(
            np.random.randn(hidden_size, hidden_size) * 0.01
        )  # New memory content

        # Biases
        self.b_z = Parameter(np.zeros((1, hidden_size)))  # Update gate
        self.b_r = Parameter(np.zeros((1, hidden_size)))  # Reset gate
        self.b_h = Parameter(np.zeros((1, hidden_size)))  # New memory content

        # Activation functions
        self.activation = ActivationFunction.get(
            activation
        )  # For the new memory content
        self.recurrent_activation = ActivationFunction.get(
            recurrent_activation
        )  # For the update and reset gates

    def forward(self, input_tensor: Tensor, hidden_state: Tensor) -> Tensor:
        z = self.recurrent_activation(
            [input_tensor @ self.W_xz + hidden_state @ self.W_hz + self.b_z]
        )
        r = self.recurrent_activation(
            [input_tensor @ self.W_xr + hidden_state @ self.W_hr + self.b_r]
        )
        h_tilde = self.activation(
            [
                input_tensor @ self.W_xh
                + (r * hidden_state) @ self.W_hh
                + self.b_h
            ]
        )
        one = Tensor.ones(shape=1)  # will broadcast
        h_new = (one - z) * h_tilde + z * hidden_state
        return h_new


class GRULayer(Layer):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = GRUCell(input_size, hidden_size)

    def forward(self, input_tensor: Tensor) -> Tensor:
        # x.shape = (sequence_length, batch_size, input_size)
        sequence_length, batch_size, _ = input_tensor.shape
        h = Tensor(
            np.zeros((batch_size, self.cell.hidden_size)), requires_grad=True
        )

        new_h_list = []
        for t in range(sequence_length):
            h = self.cell.forward(input_tensor[t], h)
            new_h_list.append(h)

        # Stack the hidden states
        hidden_states_tensor = stack(new_h_list, axis=0)

        return hidden_states_tensor


class LSTMCell(Layer):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Gates' weights and biases (input, forget, cell, output)
        self.W_xi = Parameter(
            np.random.uniform(-0.1, 0.1, (input_size, hidden_size))
        )
        self.W_hi = Parameter(
            np.random.uniform(-0.1, 0.1, (hidden_size, hidden_size))
        )
        self.b_i = Parameter(np.zeros(hidden_size))

        self.W_xf = Parameter(
            np.random.uniform(-0.1, 0.1, (input_size, hidden_size))
        )
        self.W_hf = Parameter(
            np.random.uniform(-0.1, 0.1, (hidden_size, hidden_size))
        )
        self.b_f = Parameter(np.zeros(hidden_size))

        self.W_xc = Parameter(
            np.random.uniform(-0.1, 0.1, (input_size, hidden_size))
        )
        self.W_hc = Parameter(
            np.random.uniform(-0.1, 0.1, (hidden_size, hidden_size))
        )
        self.b_c = Parameter(np.zeros(hidden_size))

        self.W_xo = Parameter(
            np.random.uniform(-0.1, 0.1, (input_size, hidden_size))
        )
        self.W_ho = Parameter(
            np.random.uniform(-0.1, 0.1, (hidden_size, hidden_size))
        )
        self.b_o = Parameter(np.zeros(hidden_size))

        # Activation functions
        self.sigmoid = ActivationFunction.get("sigmoid")
        self.tanh = ActivationFunction.get("tanh")

    def forward(self, x, h_prev, c_prev):
        i = self.sigmoid(x @ self.W_xi + h_prev @ self.W_hi + self.b_i)
        f = self.sigmoid(x @ self.W_xf + h_prev @ self.W_hf + self.b_f)
        o = self.sigmoid(x @ self.W_xo + h_prev @ self.W_ho + self.b_o)
        g = self.tanh(x @ self.W_xc + h_prev @ self.W_hc + self.b_c)
        c_new = f * c_prev + i * g
        h_new = o * self.tanh(c_new)

        return h_new, c_new


class LSTMLayer(Layer):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x):
        sequence_length, batch_size, _ = x.shape
        h = Tensor(
            np.zeros((batch_size, self.cell.hidden_size)), requires_grad=True
        )
        c = Tensor(
            np.zeros((batch_size, self.cell.hidden_size)), requires_grad=True
        )

        outputs = []
        for t in range(sequence_length):
            h, c = self.cell.forward(x[t], h, c)
            outputs.append(h)

        return stack(outputs, axis=0)
