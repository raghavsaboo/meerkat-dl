from __future__ import annotations

from typing import Union

import numpy as np
from mdl.autodiff.dcgraph import DCGraph
from mdl.autodiff.operations import ParameterOperation
from mdl.net.layer import Layer
from mdl.net.layer import Module
from mdl.net.layer import Sequence
from mdl.net.loss import Loss
from mdl.tensor import Tensor


def calculate_loss(
    component: Union[ParameterOperation, Sequence, Layer, Module],
    input_tensor: Tensor,
    target: Tensor,
    loss_fn: Loss,
) -> Union[Tensor]:
    if isinstance(component, ParameterOperation):
        output = component([input_tensor])  # type: ignore
    else:
        output = component(input_tensor)  # type: ignore

    loss = loss_fn(output, target)

    return loss


def gradient_checker(
    component: Union[ParameterOperation, Sequence, Layer, Module],
    input_tensor: Tensor,
    target: Tensor,
    loss_fn: Loss,
    epsilon=1e-7,
) -> np.float32:

    parameters = component.aggregate_parameters_as_list()

    for param in parameters:
        param.zero_grad()

    perturbed_gradients = []
    backprop_gradients = []

    graph = DCGraph()
    graph.reset_graph()

    loss = calculate_loss(component, input_tensor, target, loss_fn)
    loss.backward()

    for param in parameters:
        if param.requires_grad:
            for index in np.ndindex(param.shape):
                original_data = param.data.copy()

                backprop_gradients.append(param.grad[index])

                param.data[index] += epsilon
                pos_perturbation_loss = calculate_loss(
                    component, input_tensor, target, loss_fn
                )
                param.data = original_data
                param.data[index] -= epsilon
                neg_perturbation_loss = calculate_loss(
                    component, input_tensor, target, loss_fn
                )
                param.data = original_data

                perturbed_gradients.append(
                    (pos_perturbation_loss - neg_perturbation_loss)
                    / (2 * epsilon)
                )

            param.zero_grad()

    perturbed_gradients = np.asarray(perturbed_gradients)  # type: ignore
    backprop_gradients = np.asarray(backprop_gradients)  # type: ignore

    diff = np.linalg.norm(
        perturbed_gradients - backprop_gradients  # type: ignore
    ) / (
        np.linalg.norm(perturbed_gradients)
        + np.linalg.norm(backprop_gradients)
    )

    return diff
