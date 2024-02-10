import numpy as np
from mdl.tensor import Tensor, Parameter
from abc import ABC, abstractclassmethod

class Loss(ABC):
    
    def __call__(self, outputs, targets):
        self.batch_size = self.batch_size(outputs)
        return self.forward(outputs, targets)
    
    @staticmethod
    def batch_size(outputs):
        if outputs.shape == ():
            return 1
        return outputs.shape[0]
    
    @abstractclassmethod
    def forward(self, outputs, targets):
        raise NotImplementedError("Loss needs a forward function")
    

class MeanSquaredLoss(Loss):
    
    def forward(self, outputs, targets):
        squared_error = (outputs - targets) ** 2
        sum_squared_errors = squared_error.sum()
        return sum_squared_errors / self.batch_size
        