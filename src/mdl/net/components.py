from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List

from mdl.autodiff.operations import ParameterOperation
from mdl.tensor import Parameter
from mdl.tensor import Tensor


class Layer(ABC):
    """
    Encapsulate operations to add to a neural net module.
    """

    def __init__(self):
        self._eval: bool = False

    @property
    def eval(self) -> bool:
        return self._eval

    @eval.setter
    def eval(self, value: bool = False) -> None:
        parameters = self.aggregate_parameters_as_list()
        for parameter in parameters:
            parameter.eval = value  # type: ignore[union-attr]

        self._eval = value

    def __call__(
        self,
        input_tensor: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        return self.forward(input_tensor, *args, **kwargs)

    @abstractmethod
    def forward(
        self,
        input_tensor: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        raise NotImplementedError(
            f"Forward method not implemented for layer {self}",
        )

    def aggregate_parameters_as_dict(self) -> Dict[str, Parameter]:
        parameters = dict()
        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                parameters[name] = value
            elif isinstance(value, ParameterOperation):
                operation_params = value.aggregate_parameters_as_dict()
                for (
                    sub_name,
                    sub_param,
                ) in operation_params.items():  # type: ignore[union-attr]
                    parameters[f"{name}.{sub_name}"] = sub_param

        return parameters

    def aggregate_parameters_as_list(self) -> List[Parameter]:
        return list(self.aggregate_parameters_as_dict().values())


class Sequence(ABC):
    """
    Contains multiple feed forward layers.
    """

    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self._eval: bool = False

    def __setattr__(self, name: str, value: Any) -> None:
        raise Exception("Cannot modify or add attributes in Sequence")

    def forward(self, input_tensor: Tensor) -> Tensor:
        for layer in self.layers:
            output = layer(input_tensor)
            input_tensor = output
        return output

    def aggregate_parameters_as_dict(self) -> Dict[str, Parameter]:
        parameters = dict()
        for index, layer in enumerate(self.layers):
            name = f"SequenceLayer{index}"
            layer_parameters = layer.aggregate_parameters_as_dict()
            for (
                sub_name,
                sub_param,
            ) in layer_parameters.items():  # type: ignore[union-attr]
                parameters[f"{name}.{sub_name}"] = sub_param

        return parameters

    def aggregate_parameters_as_list(self) -> List[Parameter]:
        return list(self.aggregate_parameters_as_dict().values())

    @property
    def eval(self) -> bool:
        return self._eval

    @eval.setter
    def eval(self, value: bool = False) -> None:
        parameters = self.aggregate_parameters_as_list()
        for parameter in parameters:
            parameter.eval = value  # type: ignore[union-attr]

        self._eval = value

    def __call__(self, input_tensor: Tensor) -> Any:
        return self.forward(input_tensor)


class Module(ABC):
    """
    Consists of sequences, layers and parameters
    """

    def __init__(self):
        self._eval: bool = False

    def aggregate_parameters_as_dict(self) -> Dict[str, Parameter]:
        parameters = dict()

        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                parameters[name] = value
            elif isinstance(
                value, (Module, Layer, Sequence, ParameterOperation)
            ):
                sub_params = value.aggregate_parameters_as_dict()
                for (
                    sub_name,
                    sub_param,
                ) in sub_params.items():
                    parameters[f"{name}.{sub_name}"] = sub_param

        return parameters

    def aggregate_parameters_as_list(self) -> List[Parameter]:
        return list(self.aggregate_parameters_as_dict().values())

    @property
    def eval(self) -> bool:
        return self._eval

    @eval.setter
    def eval(self, value: bool = False) -> None:
        parameters = self.aggregate_parameters_as_list()
        for parameter in parameters:
            parameter.eval = value  # type: ignore[union-attr]

        self._eval = value

    def __call__(
        self,
        input_tensor: List[Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        return self.forward(input_tensor, *args, **kwargs)

    @abstractmethod
    def forward(
        self,
        input_tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        raise NotImplementedError(
            f"Forward method not implemented for layer {self}",
        )


# TODO: add all parameter operations as layers
# TODO: add a Module class to encapsulate all layers
