import numpy as np
from src.loss.cross_entropy_loss import CrossEntropyLoss
from src.loss.self_adaptive_training import SelfAdaptiveTraining
from src.loss.symmetric_cross_entropy_loss import SymmetricCrossEntropyLoss
from src.settings.settings import Settings
import torch


def get_criterion(criterion: str, settings: Settings, labels: np.ndarray, cls_weights: torch.Tensor | None = None) -> CrossEntropyLoss | SymmetricCrossEntropyLoss | SelfAdaptiveTraining:
    """
    Gets the criterion (could include self-adaptive training if enabled in settings)
    Args:
        criterion: the criterion to use
        settings: the settings object
        labels: the corresponding labels to all samples
        cls_weights: custom weights e.g., INS, use None to exclude
    Returns:
        the criteterion
    """
    match criterion:
        case "ce":
            criterion = CrossEntropyLoss(len(settings.dataset_labels), cls_weights=cls_weights)
        case "sce":
            criterion = SymmetricCrossEntropyLoss(settings.sce_alpha, settings.sce_beta, len(settings.dataset_labels), cls_weights=cls_weights)
        case _:
            raise RuntimeError(f"Criterion not implented: {criterion}")
    if settings.self_adaptive_training:
        criterion = SelfAdaptiveTraining(criterion, labels, settings.sat_momentum, settings.sat_start, len(settings.dataset_labels), cls_weights, label_weights = settings.sat_label_weights)
    return criterion


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")