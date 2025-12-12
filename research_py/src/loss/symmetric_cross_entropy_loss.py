import torch
from torch import nn
import torch.nn.functional as F


class SymmetricCrossEntropyLoss(nn.Module):
    """
    Acknowledgements:
        https://arxiv.org/pdf/1908.06112
    math:
        ce: log p_ij y_ij
        rce: log y_ij p_ij
        reduction(alpha ce + beta rce)
    """

    def __init__(self, alpha: float, beta: float, num_classes: int, cls_weights: torch.Tensor | None = None, reduction: str = "mean") -> None:
        """
        Initialises the symmetric cross entropy loss
        Args:
            alpha: multiplier for ce
            beta: multiplier for rce
            num_classes: the number of classes
            cls_weights: additional weight vector for handling class imbalances / importances
            reduction: the way to combine the batch-wise losses
        """
        super(SymmetricCrossEntropyLoss, self).__init__()
        self._alpha = alpha
        self._beta = beta
        self._num_classes = num_classes
        self._cls_weights = cls_weights
        assert reduction in ["mean", "sum"], f"Invalid reduction argument {reduction}"
        self._reduction = torch.mean if "mean" else torch.sum

    def forward(self, x: torch.Tensor, labels: torch.Tensor, reduction: bool = True, apply_cls_weights: bool = True, y: torch.Tensor | None = None, label_idxs: None = None, epoch: None = None) -> torch.Tensor:
        """
        Calculates the symmetric cross entropy loss
        Args:
            x: the input logits
            labels: a tensor of logits (shape n)
            reduction: whether to produce a single value or keep batch dimensions
            apply_cls_weight: whether to normalise each item in the batch w.r.t to the passed class weights
            y: pass custom targets if you do not want to use hard labels (labels one hot encoded) 
            label_idxs: **NOT USED**, an easy way to make compatible with self-adaptive training
            epoch: **NOT USED**, an easy way to make compatible with self-adaptive training
        Returns:
            torch.Tensor: the loss
        """
        eps = 1e-7
        x = F.softmax(x, dim=-1)
        x = torch.clamp(x, min=eps, max=1.0)
        if y is None:
            y = F.one_hot(labels, self._num_classes).float()
        y = torch.clamp(y, min=eps, max=1.0)
        ce = -torch.sum(torch.log(x) * y, dim=-1)
        rce = -torch.sum(x * torch.log(y), dim=-1)
        if not apply_cls_weights:
            return self._reduction(self._alpha * ce - self._beta * rce) if reduction else self._alpha * ce - self._beta * rce
        assert self._cls_weights is not None, "You need to init cls weights or use apply_cls_weights = False"
        cls_weights = torch.Tensor([self._cls_weights[l] for l in labels]).to(x.device)
        return self._reduction(cls_weights * (self._alpha * ce + self._beta * rce)) if reduction else cls_weights * (self._alpha * ce + self._beta * rce)


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")