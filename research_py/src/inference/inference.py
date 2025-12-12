import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from src.inference.predict import predict_GradCAM, predict_ScoreCAM
from src.models.efficientnet_lrcn import EfficientLRCN
from src.settings.settings import Settings
from src.utils.preprocess import pad2square, remove_black_borders, lhwc2Tensor
import torch
from torchvision.transforms import v2


def run_inference(settings: Settings, cam_name: str = "ScoreCAM", capture_interval: int = 10) -> None:
    """
    The function for running inference
    acknowledgements:
        https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html (video reading and writing)
    Args:
        settings: the settings for inference
        cam_name: the XAI mode to use either GradCAM or ScoreCAM
        capture_interval: every how many frames to use for the clip that is forward passed
    """
    print("Starting inference")
    assert cam_name in ["ScoreCAM", "GradCAM"], f"Enter a supported cam: {cam_name}"
    model_name = settings.work_model if settings.train else settings.inference_model
    save_path = os.path.join(settings.weights_path, model_name + ".pth")
    model = EfficientLRCN(settings)
    model.load_state_dict(torch.load(save_path))
    model.to(settings.train_dev)
    model.eval()
    acts = list()
    grads = list() # backbone[7][0].block[2]
    forward_hook = model.backbone[7][0].block[2].register_forward_hook( # .point_wise
        lambda module, input, output : acts.append(output))
    if cam_name == "GradCAM":
        model.rnn.train() # pytorch crashes otherwise
        backward_hook = model.backbone[7][0].block[2].register_full_backward_hook(
            lambda module, input, output : grads.append(output[0]))
    assert Path(settings.inference_path).is_dir(), "Enter a proper inference dir (connect the external ssd)"
    video_paths = os.listdir(Path(settings.inference_path))
    post_transforms = v2.Compose([v2.Normalize(mean=settings.mean, std=settings.standard_deviation, inplace=True)])
    save_dir = settings.inference_save_dir
    assert Path(save_dir).is_dir(), "set a proper inference save dir"

    for video_path in video_paths:
        cap = cv.VideoCapture(os.path.join(settings.inference_path, video_path))
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        name, _ = os.path.splitext(video_path) # add the class name
        class_name = "predict" if settings.inference_target is None else settings.dataset_labels[settings.inference_target]
        out = cv.VideoWriter(os.path.join(save_dir, "saved_" + name + "_" + cam_name + "_" + class_name + ".avi"), fourcc, 
                             float(cap.get(cv.CAP_PROP_FPS)/capture_interval), (settings.inference_save_res,  settings.inference_save_res))
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        clip = list()
        frame_idx = 0
        while True:
            ret, img = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                # skips the last frames of the video, which should be fine
                break
            frame_idx += 1
            if frame_idx % capture_interval != 0:
                continue
            img = remove_black_borders(img)
            img = pad2square(img, settings.image_size)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            clip.append(img)
            if len(clip) < settings.video_length:
                continue
            numpy_clip = np.array(clip)
            tensor_clip = lhwc2Tensor(numpy_clip, post_transforms, settings.train_dev).unsqueeze_(0) # LCHW -> NLCHW
            match cam_name:
                case "GradCAM":
                    predict_GradCAM(model, tensor_clip, numpy_clip, acts, grads, 
                                    out, settings.dataset_labels, settings.inference_save_res, 
                                    target=settings.inference_target)
                case "ScoreCAM":
                    predict_ScoreCAM(model, tensor_clip, numpy_clip, acts, 
                                        out, settings.dataset_labels, settings.inference_save_res, 
                                        dev=settings.train_dev, target=settings.inference_target)
            clip.clear()
        cap.release()
        out.release()
    forward_hook.remove()
    if cam_name == "GradCAM":
        backward_hook.remove()


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")