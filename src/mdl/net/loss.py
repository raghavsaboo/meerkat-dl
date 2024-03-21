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


class CrossEntropyLoss(Loss):
    def forward(self, outputs, targets):
        exp_outputs = outputs.exp()
        probs = exp_outputs / exp_outputs.sum(axis=1, keepdims=True)
        log_probs = -(
            probs[range(self.batch_size.data), targets.data.astype(int)]
        ).log()
        loss = log_probs.sum() / self.batch_size
        return loss


class BinaryCrossEntropyLoss(Loss):
    def forward(self, outputs, targets):
        probs = (-outputs).exp() + 1
        probs = probs.pow(-1)
        log_probs = targets * probs.log() + (1 - targets) * (1 - probs).log()
        loss = -log_probs.mean()
        return loss


class HingeLoss(Loss):
    def forward(self, outputs, targets):
        num_samples = outputs.shape[0]
        correct_class_scores = outputs[
            range(num_samples), targets.data.astype(int)
        ]
        margins = (
            (outputs - correct_class_scores.reshape(num_samples, 1) + 1.0)
        ).max(Tensor(0.0))
        margins[range(num_samples), targets.data.astype(int)] = Tensor(0.0)
        loss = margins.sum(axis=1).mean()
        return loss


class KLDivergenceLoss(Loss):
    def forward(self, outputs, targets):
        outputs_probs = outputs / outputs.sum(axis=1, keepdims=True)
        targets_probs = targets / targets.sum(axis=1, keepdims=True)
        kl_div = (targets_probs * (targets_probs / outputs_probs).log()).sum(
            axis=1
        )
        loss = kl_div.mean()
        return loss


class DiceLoss(Loss):
    def forward(self, outputs, targets):
        smooth = Tensor(1e-6)
        outputs = outputs.flatten()
        targets = targets.flatten()
        intersection = (outputs * targets).sum()
        dice_coeff = (Tensor(2.0) * intersection + smooth) / (
            outputs.sum() + targets.sum() + smooth
        )
        loss = Tensor(1.0) - dice_coeff
        return loss
