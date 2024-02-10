from abc import ABC, abstractclassmethod
from mdl.tensor import Tensor, Parameter
from mdl.autodiff.operations import Operation, ParameterOperation
from mdl.autodiff.dcgraph import DCGraph
from typing import List

class Optimizer(ABC):
    
    global_dc_graph = DCGraph()
    
    def __init__(self, parameters: List[Parameter], learning_rate: float):
        self.parameters = parameters
        self.learning_rate = learning_rate
        
    def zero_grad(self, only_parameters: bool = False):
        
        if only_parameters:
            for param in self.parameters:
                param.zero_grad()
                
        else:
            self.global_dc_graph.zero_grad()
            
    @abstractclassmethod
    def step(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Step method not implemented for Optimizer {self}",
        )
        
class GradientDescent(Optimizer):
    
    def __init__(self, params, learning_rate):
        super.__init__(params, learning_rate)
        
    def step(self):
        for param in self.parameters:
            if param.requires_grad:
                param.data -= (self.lr * param.grad)
                
    