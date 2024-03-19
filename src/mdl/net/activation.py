from __future__ import annotations

from typing import List

import numpy as np
from mdl.autodiff.operations import ParameterOperation
from mdl.tensor import Tensor
from enum import Enum, auto


class Activation(Enum):
    TANH = auto()
    RELU = auto()
    SIGMOID = auto()
    LEAKY_RELU = auto()
    SWISH = auto()
    GELU = auto()
    MISH = auto()


class ActivationFunction:
    @staticmethod
    def get(activation_type: Activation) -> ParameterOperation:
        activation_map = {
            Activation.TANH: Tanh,
            Activation.RELU: ReLU,
            Activation.SIGMOID: Sigmoid,
            Activation.LEAKY_RELU: LeakyReLU,
            Activation.SWISH: Swish,
            Activation.GELU: GELU,
            Activation.MISH: Mish,
        }

        activation_class = activation_map.get(activation_type)
        if not activation_class:
            raise ValueError(f"Unsupported activation type: {activation_type}")

        return activation_class()  # Instantiate the activation class


class Sigmoid(ParameterOperation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(1 / (1 + np.exp(-a.data)), self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = (
            lambda output_grad: output_grad
            * (1 / (1 + np.exp(-a.data)))
            * (1 - 1 / (1 + np.exp(-a.data)))
        )


class ReLU(ParameterOperation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(np.maximum(0, a.data), self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad * (a.data > 0).astype(
            float
        )


class Tanh(ParameterOperation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(np.tanh(a.data), self.requires_grad)

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad * (
            1 - np.tanh(a.data) ** 2
        )


class LeakyReLU(ParameterOperation):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(
            np.where(a.data > 0, a.data, a.data * self.negative_slope),
            self.requires_grad,
        )
        self.global_dc_graph.add_edge(result, input_tensors)
        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        a.grad_fn = lambda output_grad: output_grad * np.where(
            a.data > 0, 1, self.negative_slope
        )


class Softmax(ParameterOperation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        exp_values = np.exp(a.data - np.max(a.data, axis=-1, keepdims=True))
        result = Tensor(
            exp_values / np.sum(exp_values, axis=-1, keepdims=True),
            self.requires_grad,
        )
        self.global_dc_graph.add_edge(result, input_tensors)
        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        softmax_output = self._forward(input_tensors).data
        a.grad_fn = lambda output_grad: output_grad * (
            softmax_output * (1 - softmax_output)
        )


class Swish(ParameterOperation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(
            a.data * (1 / (1 + np.exp(-a.data))), self.requires_grad
        )

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        sigmoid_a = 1 / (1 + np.exp(-a.data))
        a.grad_fn = lambda output_grad: output_grad * (
            sigmoid_a + a.data * sigmoid_a * (1 - sigmoid_a)
        )


class GELU(ParameterOperation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(
            0.5
            * a.data
            * (
                1
                + np.tanh(
                    np.sqrt(2 / np.pi)
                    * (a.data + 0.044715 * np.power(a.data, 3))
                )
            ),
            self.requires_grad,
        )

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        cdf = 0.5 * (
            1
            + np.tanh(
                np.sqrt(2 / np.pi) * (a.data + 0.044715 * np.power(a.data, 3))
            )
        )
        a.grad_fn = lambda output_grad: output_grad * (
            cdf
            + a.data * np.exp(-0.5 * np.power(a.data, 2)) / np.sqrt(2 * np.pi)
        )


class Mish(ParameterOperation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        result = Tensor(
            a.data * np.tanh(np.log(1 + np.exp(a.data))), self.requires_grad
        )

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        exp_at = np.exp(a.data)
        exp_at1 = np.exp(2 * a.data)
        a.grad_fn = lambda output_grad: output_grad * (
            exp_at
            * (4 * exp_at + 4 * a.data + exp_at1 + 4)
            / (exp_at + 2 + exp_at1) ** 2
        )


class SELU(ParameterOperation):
    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        if len(input_tensors) != 1:
            raise ValueError(
                f"Expect 1 input tensor but got {len(input_tensors)}"
            )

        a = input_tensors[0]
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        result = Tensor(
            scale
            * np.where(a.data >= 0, a.data, alpha * (np.exp(a.data) - 1)),
            self.requires_grad,
        )

        self.global_dc_graph.add_edge(result, input_tensors)

        result.backward_fn = self.backward
        result.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return result

    def backward(self, input_tensors: List[Tensor]) -> None:
        a = input_tensors[0]
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        grad = np.where(a.data >= 0, scale, alpha * scale * np.exp(a.data))
        a.grad_fn = lambda output_grad: output_grad * grad
