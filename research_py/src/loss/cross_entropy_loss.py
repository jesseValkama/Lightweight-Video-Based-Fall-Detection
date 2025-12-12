import torch
from torch import nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    A custom version of the cross entropy loss, to make it compatible with self-adaptive training and TRADES
    math (see __init__):
        l = sum log p_ij y_ij
        reduction(cls_weights l) 
    """

    def __init__(self, num_classes: int, cls_weights: torch.Tensor | None = None, reduction: str = "mean") -> None:
        """
        Initialises the cross entropy loss
        Args:
            num_classes: the number of classes
            cls_weights: weights for balancing dataset imbalances
            reduction: the way to combine batch-wise losses
        """
        super(CrossEntropyLoss, self).__init__()
        self._num_classes = num_classes
        self._cls_weights = cls_weights
        assert reduction in ["mean", "sum"], f"Invalid reduction argument {reduction}, disabling reduction happens in the forward method"
        self._reduction = torch.mean if "mean" else torch.sum

    def forward(self, x: torch.Tensor, labels: torch.Tensor, reduction: bool = True, apply_cls_weights: bool = True, y: torch.Tensor | None = None, label_idxs: None = None, epoch: None = None) -> torch.Tensor:
        """
        Calculates the cross entropy loss
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
        x = torch.clamp(x, min=eps)
        if y is None:
            y = F.one_hot(labels, self._num_classes).float()
        y = torch.clamp(y, min=eps)
        ce = -torch.sum(torch.log(x) * y, dim=-1)
        if not apply_cls_weights:
            return self._reduction(ce) if reduction else ce
        assert self._cls_weights is not None, "You need to init cls weights or use apply_cls_weights = False"
        cls_weights = torch.Tensor([self._cls_weights[l] for l in labels]).to(x.device)
        return self._reduction(cls_weights * ce) if reduction else cls_weights * ce
    

if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")