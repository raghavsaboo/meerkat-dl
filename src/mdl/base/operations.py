import numpy as np
from abc import ABC, abstractmethod
from mdl.tensor import Tensor
from numpy import ndarray
from typing import List

class Operation(ABC):

    @property
    def requires_grad(self):
        pass
    
    @requires_grad.setter
    @abstractmethod
    def requires_grad(self, input_tensors: List[Tensor]) -> None:
        self.requires_grad = any([tensor.requires_grad for tensor  in input_tensors]) 

    @classmethod
    @abstractmethod
    def forward(self, input_tensors):
        raise NotImplementedError(f"Forward method not implemented for operator {self}")
    
    @abstractmethod
    def _input_gradient(self):
        raise NotImplementedError(f"Input gradient not defined for operator {self}")
    
    @classmethod
    @abstractmethod
    def backward(self):
        raise NotImplementedError(f"Backward pass not implemented for operator {self}")
    

class Add(Operation):
    
    def forward(self, a, b):
        self.requires_grad([a.requires_grad, b.requires_grad])
        result = Tensor(a.data + b.data, self.requires_grad)
        
        return result
    
    