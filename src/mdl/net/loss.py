from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from mdl.tensor import Tensor

class Loss(ABC):

    def __call__(self, outputs, targets):
        self.batch_size = self.calc_batch_size(outputs)
        return self.forward(outputs, targets)

    @staticmethod
    def calc_batch_size(outputs):
        if outputs.shape == ():
            return Tensor(1)
        return Tensor(outputs.shape[0])

    @abstractmethod
    def forward(self, outputs, targets):
        raise NotImplementedError("Loss needs a forward function")


class MeanSquaredLoss(Loss):

    def forward(self, outputs, targets):
        squared_error = (outputs - targets) ** 2
        sum_squared_errors = squared_error.sum()
        return sum_squared_errors / self.batch_size
