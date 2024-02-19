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
    def forward(self, outputs, targets) -> Tensor:
        raise NotImplementedError("Loss needs a forward function")


class MeanSquaredErrorLoss(Loss):

    def forward(self, outputs, targets):
        diff = outputs - targets
        squared_error = (diff) ** 2
        sum_squared_errors = squared_error.sum()
        loss = sum_squared_errors / (self.batch_size * Tensor(2.0))
        return loss
