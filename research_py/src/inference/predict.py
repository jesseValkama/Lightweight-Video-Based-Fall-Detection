import cv2 as cv
import numpy as np
from src.inference.write import write_video
from src.models.efficientnet_lrcn import EfficientLRCN
import torch
from torch.amp import autocast
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import List, Tuple


def predict_GradCAM(model: EfficientLRCN, clip: torch.Tensor, rgb_clip: np.ndarray, acts: List, grads: List, out: cv.VideoWriter, 
            dataset_labels: List[str], inference_resize: int, target: int | None = None) -> None:
    """
    GradCAM modified to work with vidoe data
    Acknowledgements:
        https://arxiv.org/pdf/1610.02391
    Args:
        model: the model
        clip: the clip as tensor in LCHW
        rgb_clip: the clip as np.ndarray in LHWC
        acts: list to store activations
        grads: list to store gradients
        out: the video writer
        dataset_labels: list of labels as str
        inference_resize: the output size for the written vid
        taget: the class to visualise or None to visualise predictions
    """
    with autocast(device_type="cuda"):
        logits = model(clip)
    logit, idx = torch.max(logits, dim=1) if target is None else logits[:, target], target
    logit.backward()
    alpha = torch.mean(grads[0], dim=(2,3))
    alpha = alpha[:, :, None, None]
    grad_cam = F.relu(torch.sum(acts[0].cpu() * alpha.cpu(), dim=1))
    acts.clear()
    grads.clear()
    write_video(rgb_clip, grad_cam, out, logits, idx, inference_resize, dataset_labels)
    

@torch.no_grad
def predict_ScoreCAM(model: EfficientLRCN, clip: torch.Tensor, rgb_clip: np.ndarray, acts: List, out: cv.VideoWriter, 
            dataset_labels: List[str], inference_resize: int, target: int | None = None, BATCH_SIZE: int = 30, dev: str = "cuda:0") -> None:
    """
    ScoreCAM modified to work with video data (could be bugged as the activations are spiral shaped)
    Acknowledgements
        https://arxiv.org/pdf/1910.01279 (scorecam official paper)
        https://github.com/jacobgil/pytorch-grad-cam/tree/master (was helpful when trying to understand the paper)
        https://github.com/haofanwang/Score-CAM/blob/master/cam/scorecam.py#L52 (official impl)
    Args:
        model: the model
        clip: the clip as tensor in LCHW
        rgb_clip: the clip as np.ndarray in LHWC
        acts: list to store activations
        out: the video writer
        dataset_labels: list of labels as str
        inference_resize: the output size for the written vid
        taget: the class to visualise or None to visualise predictions
        batch_size: batch_size for batching the channels for scorecam
        dev: inference device 
    """
    with autocast(device_type="cuda"):
        logits = model(clip)
    A_k = acts[0]
    acts.clear()
    idx = torch.max(logits, dim=1)[1] if target is None else target

    M = F.interpolate(A_k, size=rgb_clip.shape[-3:-1], mode="bilinear")
    M = s(M, dims=(2,3)) # NCHW
    clip = clip.to(torch.float16).squeeze_(0)[:, None, :, :, :]
    S_k = torch.Tensor([]).to(dev)
    for i in range(0, A_k.size(1), BATCH_SIZE):
        M_k = M[:, i:i+BATCH_SIZE, None, :, :] * clip # (LCHW -> LC1HW) * (L3HW -> L13HW) -> LC3HW
        M_k = M_k.transpose(0, 1).contiguous() # LC3HW -> CL3HW : C works as the batch size
        with autocast(device_type="cuda"):
            outputs = model(M_k)[:, idx]
        S_k = torch.cat((S_k, outputs.view(-1)), dim=0)
        acts.clear()
    alpha_k = F.softmax(S_k.unsqueeze_(0), dim=1) # C -> 1C, no need for normalisation as the official impl doesn't use it
    alpha_k = alpha_k[:, :, None, None] # 1C11
    score_cam = F.relu(torch.sum(alpha_k.cpu() * A_k.cpu(), dim=1))
    write_video(rgb_clip, score_cam, out, logits, idx, inference_resize, dataset_labels)   


def s(M: torch.Tensor, dims: Tuple[int, int]) -> torch.Tensor:
    """
    Normalisation function for scorecam:
        https://arxiv.org/pdf/1910.01279
    Args:
        M: the upsampled activations
        dims: the dims to take the max and min over
    Returns:
        torch.Tensor: the normalised upsampled activations
    """
    maxs, mins = torch.amax(M, dim=dims), torch.amin(M, dim=dims)
    maxs, mins = maxs[:, :, None, None], mins[:, :, None, None] # NC -> NC11
    return (M - mins) / (maxs - mins + 1e-8)


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py")