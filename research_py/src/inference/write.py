import cv2 as cv
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import torch.nn.functional as F
from typing import List


def write_video(rgb_clip: np.ndarray, cam: torch.Tensor, out: cv.VideoWriter, logits: torch.Tensor, idx: torch.Tensor, inference_resize: int, dataset_labels: List[str]) -> None:
    """
    Function for writing video data
    Args:
        rgb_clip: the clip as a np.array, not normalised
        cam: the grayscale heatmap from a cam
        out: the video writer
        idx: the index for the class to visualise
        inference_resize: the resolution for the written video
        dataset_labels: the list of dataset labels
    """
    rgb_clip = rgb_clip / 255
    cam = cam.detach().numpy().astype(np.float32) # opencv breaks with fp16
    for i in range(len(cam)):
        heatmap = show_cam_on_image(rgb_clip[i], cv.resize(cam[i], rgb_clip[i][:,:,0].shape), use_rgb=True)
        heatmap = cv.resize(heatmap, (inference_resize, inference_resize)) # cv.putText quality
        heatmap = cv.cvtColor(heatmap, cv.COLOR_RGB2BGR)
        confs = F.softmax(logits, dim=1).view(-1)
        cv.putText(heatmap, f"{dataset_labels[idx]} : {confs[idx].item():.2f}", (30, 40), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2, cv.LINE_AA)
        out.write(heatmap)


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")