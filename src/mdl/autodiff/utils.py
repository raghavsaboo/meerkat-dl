from __future__ import annotations

from typing import Union

import numpy as np
from mdl.autodiff.dcgraph import DCGraph
from mdl.autodiff.operations import ParameterOperation
from mdl.net.components import Layer
from mdl.net.components import Module
from mdl.net.components import Sequence
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
    epsilon=1e-12,
) -> np.float64:

    parameters = component.aggregate_parameters_as_list()
    print(f"parameters: {parameters}")

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
                backprop_gradients.append(param.grad[index])
                param.data[index] += epsilon
                pos_perturbation_loss = calculate_loss(
                    component, input_tensor, target, loss_fn
                ).data
                param.data[index] -= 2 * epsilon
                neg_perturbation_loss = calculate_loss(
                    component, input_tensor, target, loss_fn
                ).data
                param.data[index] += epsilon

                perturbed_gradients.append(
                    (pos_perturbation_loss - neg_perturbation_loss)
                    / (2 * epsilon)
                )

        param.zero_grad()

    perturbed_gradients = np.asarray(perturbed_gradients)  # type: ignore
    backprop_gradients = np.asarray(backprop_gradients)  # type: ignore

    print(f"perturbed_gradients: {perturbed_gradients}")
    print(f"backprop_gradients: {backprop_gradients}")

    diff = np.linalg.norm(
        perturbed_gradients - backprop_gradients  # type: ignore
    ) / (
        np.linalg.norm(perturbed_gradients)
        + np.linalg.norm(backprop_gradients)
    )

    return diff
