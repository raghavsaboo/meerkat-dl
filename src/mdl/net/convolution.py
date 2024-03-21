from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Generator
from typing import Any
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from mdl.autodiff.operations import Operation
from mdl.tensor import Parameter
from mdl.tensor import Tensor
from mdl.utilities import unbroadcast


class Convolution(Operation):
    """Base class for Convolution and Pooling operations

    Parameters:
      padding (int): Padding value to be applied
      stride (int): Stride to be taken
    """

    def __init__(self, padding=0, stride=1):
        super().__init__()
        self.validate_input_tensors(padding, stride)
        self.padding = padding
        self.stride = stride

    @staticmethod
    def validate_parameters(padding: int, stride: int):
        if stride < 1:
            raise ValueError("Stride must be at least 1")
        if padding < 0:
            raise ValueError("Padding cannot be negative")

    def _get_tensor_segments(
        self, padded_data: np.ndarray, kernel_shape: Tuple[int, ...]
    ) -> Generator[Tuple[np.ndarray, slice, slice], None, None]:
        _, _, H, W = padded_data.shape
        _, _, KH, KW = kernel_shape
        for i in range(0, H - KH + 1, self.stride):
            for j in range(0, W - KW + 1, self.stride):
                row_slice = slice(i, i + KH)
                col_slice = slice(j, j + KW)
                yield padded_data[
                    :, :, row_slice, col_slice
                ], row_slice, col_slice

    def get_output_shape(
        self,
        inputs_shape: Tuple[int, ...],
        kernel_shape: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        N, C, H, W = inputs_shape
        _, _, KH, KW = kernel_shape
        H_out = math.floor(((H + 2 * self.padding - KH) / self.stride) + 1)
        W_out = math.floor(((W + 2 * self.padding - KW) / self.stride) + 1)
        return (N, C, H_out, W_out)

    def pad(self, data: np.ndarray):
        return np.pad(
            data,
            pad_width=(
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            ),
            mode="constant",
            constant_values=0,
        )

    def unpad(self, padded_data: np.ndarray):
        if self.padding > 0:
            return padded_data[
                :,
                :,
                self.padding : -self.padding,
                self.padding : -self.padding,
            ]
        return padded_data

    def segment_iterator(
        self,
        padded_inputs: np.ndarray,
        kernel_shape: Tuple[int, ...],
        *args,
    ):
        segments = self._get_tensor_segments(padded_inputs, kernel_shape)
        return zip(segments, *args)

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


class Conv2D(Convolution):

    def _forward(self, input_tensors: List[Tensor | Parameter]) -> Tensor:
        input_tensor, kernel, bias = input_tensors

        assert (
            len(input_tensor.shape) != 3
        ), "Need to pass 3D tensor, with shape (batch size, m, n)"

        output = Tensor(
            np.empty(
                (
                    input_tensor.shape[0],
                    *self.get_output_shape(input_tensor.shape, kernel.shape),
                )
            ),
            self.requires_grad,
        )

        padded_inputs = self.pad(input_tensor.data)

        for (segment, _, _), (i, j) in self.segment_iterator(
            padded_inputs, kernel.shape, np.ndindex(output.shape[-2:])
        ):
            cross_corr = (
                np.sum((segment * kernel.data), axis=(1, 2)) + bias.data
            )
            output[:, i, j] = cross_corr

        self.global_dc_graph.add_edge(output, input_tensors)

        output.backward_fn = self.backward
        output.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return output

    def backward(self, input_tensors: List[Tensor]):
        input_tensor, kernel, bias = input_tensors

        padded_inputs = self.pad(input_tensor.data)

        def inputs_grad_fn(output_grad):
            input_tensor_grad = np.zeros(padded_inputs.shape)

            for (segment, row_slice, col_slice), (i, j) in self.segment_iterator(
                padded_inputs, kernel.shape, np.ndindex(output_grad.shape[-2:])
            ):
                sliced_output_grad = output_grad[:, i, j]
                sum_grad = np.ones(segment.shape) * sliced_output_grad.reshape(
                    sliced_output_grad.size, 1, 1
                )
                segment_grad = kernel.data * sum_grad
                input_tensor_grad[:, row_slice, col_slice] += segment_grad

            unpadded_local_grad = self.unpad(input_tensor_grad)
            return unpadded_local_grad

        def kernel_grad_fn(output_grad):
            kernel_grads = np.zeros(kernel.shape)

            for (segment, row_slice, col_slice), (i, j) in self.segment_iterator(
                padded_inputs, kernel.shape, np.ndindex(output_grad.shape[-2:])
            ):
                sliced_output_grad = output_grad[:, i, j]
                sum_grad = np.ones(segment.shape) * sliced_output_grad.reshape(
                    sliced_output_grad.size, 1, 1
                )
                kernel_grad = unbroadcast(
                    segment * sum_grad, kernel.shape, segment.shape
                )
                kernel_grads += kernel_grad

            return kernel_grads

        def bias_grad_fn(output_grad):
            return np.sum(output_grad)

        input_tensor.grad_fn = inputs_grad_fn
        kernel.grad_fn = kernel_grad_fn
        bias.grad_fn = bias_grad_fn


def conv2d(input_tensors, padding, stride):
    return Conv2D(padding, stride).forward(input_tensors)


class Conv3D(Convolution):

    def _forward(self, input_tensors: List[Tensor]) -> Tensor:
        input_tensor, kernel, bias = input_tensors

        assert (
            len(input_tensor.shape) != 4
        ), "Need to pass 3D tensor, with shape (batch size, m, n)"

        output = Tensor(
            np.empty(
                (
                    input_tensor.shape[0],
                    *self.get_output_shape(input_tensor.shape, kernel.shape),
                )
            ),
            self.requires_grad,
        )
        padded_inputs = self.pad(input_tensor.data)

        for (segment, _, _), (i, j) in self.segment_iterator(
            padded_inputs, kernel.shape, np.ndindex(output.shape[-2:])
        ):
            expanded_segment = np.expand_dims(segment, axis=1)
            cross_corr = (
                np.sum((expanded_segment * kernel.data), axis=(2, 3, 4))
                + bias.data
            )
            output[:, :, i, j] = cross_corr

        self.global_dc_graph.add_edge(output, input_tensors)

        output.backward_fn = self.backward
        output.parent_broadcast_shape = self.input_broadcast_shape(
            input_tensors
        )

        return output

    def backward(self, input_tensors: List[Tensor]):
        input_tensor, kernel, bias = input_tensors

        padded_inputs = self.pad(input_tensor.data)

        def input_grad_fn(output_grad):
            input_tensor_grad = np.zeros(padded_inputs.shape)

            for (segment, row_slice, col_slice), (i, j) in self.segment_iterator(
                padded_inputs, kernel.shape, np.ndindex(output_grad.shape[-2:])
            ):
                expanded_segment = np.expand_dims(segment, axis=1)
                sliced_output_grad = output_grad[:, :, i, j]
                sliced_output_grad = sliced_output_grad.reshape(
                    *sliced_output_grad.shape, 1, 1, 1
                )
                sum_grad = np.ones(expanded_segment.shape) * sliced_output_grad
                segment_grad = np.sum(kernel.data * sum_grad, axis=1)
                input_tensor_grad[:, :, row_slice, col_slice] += segment_grad

            unpadded_local_grad = self.unpad(input_tensor_grad)
            return unpadded_local_grad

        def kernel_grad_fn(output_grad):
            kernel_grads = np.zeros(kernel.shape)

            for (segment, row_slice, col_slice), (i, j) in self.segment_iterator(
                padded_inputs, kernel.shape, np.ndindex(output_grad[-2:])
            ):
                expanded_segment = np.expand_dims(segment, 1)
                sliced_output_grad = output_grad[:, :, i, j]
                sliced_output_grad = sliced_output_grad.reshape(
                    *sliced_output_grad.shape, 1, 1, 1
                )
                sum_grad = np.ones(expanded_segment.shape) * sliced_output_grad
                kernel_grad = unbroadcast(
                    expanded_segment * sum_grad, kernel.shape, segment.shape
                )
                kernel_grads += kernel_grad

            return kernel_grads

        def bias_grad_fn(output_grad):
            grad = np.sum(output_grad, axis=0)
            grad = np.sum(grad, axis=2, keepdims=True)
            grad = np.sum(grad, axis=1, keepdims=True)
            return grad

        input_tensor.grad_fn = input_grad_fn
        kernel.grad_fn = kernel_grad_fn
        bias.grad_fn = bias_grad_fn


def conv3d(input_tensors, padding, stride):
    return Conv3D(padding, stride).forward(input_tensors)
