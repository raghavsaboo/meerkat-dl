from __future__ import annotations
import numpy as np
from typing import Union
from mdl.base.operations import Add
from mdl.autodiff.dcgraph import DCGraph

TensorDataTypes = Union[float, int, list, np.ndarray]

class Tensor:
    
    global_dc_graph = DCGraph()
    
    def __init__(self, data: TensorDataTypes, requires_grad: bool = False) -> None:
        self._data = self._convert_to_ndarray(data)
        self._requires_grad = requires_grad
        
        if self._requires_grad:
            self.set_gradients_to_zero()
        
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data: TensorDataTypes):
        self._data = self._convert_to_ndarray(data)
        # changing the data means that the current gradient
        # is invalid
        self._grad = None
        
    @property
    def requires_grad(self):
        return self._requires_grad
    
    @requires_grad.setter
    def requires_grad(self, requires_grad: bool = False):
        self._requires_grad = requires_grad
        # resetting grads after changing requesting grads
        self.set_gradients_to_zero()
        
    def set_gradients_to_zero(self):
        self.grad = np.zeros_like(self._data)
    
    @staticmethod
    def _convert_to_ndarray(data: TensorDataTypes) -> np.ndarray:
        if type(data) not in {float, int, np.ndarray}:
            ValueError("Incompatible type for `data`. Expect float, int or numpy array.")
        
        if type(data) == np.ndarray:
            return data
        else:
            return np.array(data)
            
        
    def to_list(self):
        return self._data.tolist()
    
    def to_array(self):
        return self._data
    
    def __add__(self, b: TensorDataTypes) -> Tensor:
        addition_op = Add()
        return addition_op.forward(self, b)
        
    
        
        