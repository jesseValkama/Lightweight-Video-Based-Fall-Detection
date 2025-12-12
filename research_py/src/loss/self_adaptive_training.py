import numpy as np
from src.loss.cross_entropy_loss import CrossEntropyLoss
from src.loss.symmetric_cross_entropy_loss import SymmetricCrossEntropyLoss
import torch
from torch import nn
import torch.nn.functional as F


class SelfAdaptiveTraining(nn.Module):
    """
    Acknowledgements:
        https://arxiv.org/abs/2101.08732 (official paper)
        https://github.com/LayneH/self-adaptive-training (official repo)
    math (see __init__) or simply the official paper:
    if e_cur > e_s 
        t_i <- alpha * t_i + (1 - alpha) * p_i
        w_i = max_j t_ij
        w_i *= N / sum w_i
        reduction(w_cls w_i criterion(t_ij, p_ij))
        added term: cls_weights (w_cls) which balances the inclass balances, if enabled
    otherwise:
        criterion
    """
    
    def __init__(self, criterion: CrossEntropyLoss | SymmetricCrossEntropyLoss, labels: np.ndarray, momentum: float, start_epoch: int, num_classes: int, cls_weights: torch.Tensor | None = None, label_weights: bool = True, reduction: str = "mean", dev: str = "cuda:0") -> None:
        """
        Method to initialise the self-adaptive training
        Args:
            criterion: needs to be a custom one (see CrossEntropyLoss / SymmetricCrossEntropyLoss)
            labels: **not for the current timestep** they are all labels from the entire training data
            momentum: alpha
            start_epoch: e_s
            num_classes: the number of classes
            cls_weights: additional per-class weights
            label_weights: whether to use the label weights as proposed in the paper (w_i)
            reduction: combine the batch-wise losses
            dev: the device used for training
        """
        super(SelfAdaptiveTraining, self).__init__()
        self._criterion = criterion
        self._cls_weights = cls_weights
        self._momentum = momentum
        self._start_epoch = start_epoch
        self._adaptive_labels = F.one_hot(torch.Tensor(labels).long(), num_classes).float().to(dev)
        self._num_classes = num_classes
        self._label_weights = label_weights
        assert reduction in ["mean", "sum"], f"Invalid reduction argument {reduction}"
        self._reduction = torch.mean if reduction == "mean" else torch.sum

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, label_idxs: torch.Tensor, epoch: int, reduction: bool = False, apply_cls_weights: bool = False) -> torch.Tensor:
        """
        Calculates the loss according to the self-adaptive training math
        Args:
            logits: the input logits
            labels: the correct classes for the logits
            label_idxs: the indexes for the labels out of all training labels
            epoch: the current epoch or 0 to not use self-adaptive training (e.g., validation)
            reduction: **NOT USED** to make compatible with ce / sce
            apply_cls_weights: **NOT USED** to make compatible with ce / sce
        Returns:
            Tensor: The loss
        """
        if epoch < self._start_epoch:
            return self._criterion(logits, labels, apply_cls_weights=apply_cls_weights)
        self._adaptive_labels[label_idxs] = self._momentum * self._adaptive_labels[label_idxs] + (1 - self._momentum) * F.softmax(logits.detach(), dim=1)
        weights, _ = torch.max(self._adaptive_labels[label_idxs], dim=1)
        weights = weights * logits.shape[0] / torch.sum(weights) if self._label_weights else 1.0
        loss = self._criterion(logits, labels, reduction=False, apply_cls_weights=False, y=self._adaptive_labels[label_idxs])
        if self._cls_weights is None:
            return self._reduction(loss * weights)
        cls_weights = torch.Tensor([self._cls_weights[l] for l in labels]).to(logits.device)
        return self._reduction(cls_weights * (loss * weights))


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")