import albumentations as A
import av
import cv2 as cv
import numpy as np
import os
from pathlib import Path
from src.settings import Settings
import torch
from torchvision.transforms import v2
from typing import Tuple, Dict


def get_omnifall_datasets(ds_info: Dict, settings: Settings) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Function for settings up Omnifall datasets
    Setup transforms in this function
    Args:
        ds_info: dataset info loaded with load_omnifall_info
        settings: Settings object
    Returns:
        Tuple of datasets: (train, val, test)
    """
    pre_transforms = A.ReplayCompose([
        A.Resize(width=settings.image_size, height=settings.image_size)
    ])
    aug_transforms = A.ReplayCompose([
        A.HorizontalFlip(),
        A.ISONoise(p=0.3),
        A.RandomGamma(p=0.3)
    ])
    # aug_transforms2 were used in early experiments, not a part of ablation studies
    aug_transforms2 = A.ReplayCompose([
        A.HorizontalFlip(),
        A.VerticalFlip(p=0.3),
        A.Transpose(p=0.3),
        A.ISONoise(p=0.3),
        A.RandomGamma(p=0.3),
        A.MotionBlur(p=0.2)
    ])
    post_transforms = v2.Compose([
        v2.Normalize(mean=settings.mean, std=settings.standard_deviation, inplace=True)  
    ])
    train = Omnifall(ds_info["train"], settings, pre_transforms, post_transforms, aug_transforms=aug_transforms)
    val = Omnifall(ds_info["validation"], settings, pre_transforms, post_transforms)
    test = Omnifall(ds_info["test"], settings, pre_transforms, post_transforms)
    return train, val, test


class Omnifall(torch.utils.data.Dataset):
    """
    Class for handling Omnifall
    """
    
    def __init__(self, ds_info: dict, settings: Settings, pre_transforms: A.ReplayCompose, post_transforms: v2.Compose, aug_transforms: A.ReplayCompose | None = None) -> None:
        """
        Initialises the dataset object
        Args:
            ds_info: ds_info from load_omnifall_info for the corresponding set: train, val, or test
            settings: Settings object
            pre_transforms: **NOT USED**
            post_transforms: v2.Compose for normalising the tensor
            aug_transforms: Albumentation ReplayCompose for data augmentations or none to skip
        """
        assert isinstance(pre_transforms, A.ReplayCompose) and isinstance(aug_transforms, (A.ReplayCompose, type(None))) and isinstance(post_transforms, v2.Compose)
        self._video_paths = ds_info["paths"]
        self._video_datasets =  ds_info["datasets"]
        self._video_times = ds_info["times"] 
        self._video_labels = ds_info["labels"]
        self._settings = settings
        self._video_len = settings.video_length
        self._pre_transforms = pre_transforms
        self._post_transforms = post_transforms
        self._aug_transforms = aug_transforms
        assert len(self._video_paths) == len(self._video_labels)

    def __len__(self) -> int:
        """
        returns the number of samples
        Returns:
          int: the number of samples
        """
        return len(self._video_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        gets the datasamples from index idx
        sample is normalised according to post transforms
        potentially augmentations if enabled
        Args:
            idx: the index
        Returns:
            Tuple: a pair of a sample and label
        """
        ext = self._get_ext(idx)
        video_path = Path(os.path.join(self._settings.disk(self._video_datasets[idx]), self._settings.dataset_path, self._video_datasets[idx], self._video_paths[idx] + ext))
        assert video_path.is_file(), f"path to video is invalid {video_path}"
        clip = self._load_video(video_path, self._video_times[idx], self._video_datasets[idx])
        if self._aug_transforms:
            clip = self._apply_transforms(clip, self._aug_transforms)

        clip = torch.Tensor(clip) 
        clip = clip.permute([0, 3, 1, 2]).contiguous().to(torch.float)
        clip.div_(255.0)
        clip = self._post_transforms(clip)
        label = self._video_labels[idx]
        label = torch.Tensor([label]).to(torch.long)
        return clip, label, idx
    
    def _apply_transforms(self, clip: np.ndarray, transforms: A.Compose) -> torch.Tensor:
        """
        Method for applying pre and aug transforms since Albumentations don't work
        naively with videos
        Args:
            clip: the clip as an array
            transforms: the augmentations as albumentations
        Returns:
            np.array: the augmented clip 
        """
        n = len(clip)
        t = transforms(image=clip[0])
        replay = t["replay"]
        transformed = [t["image"]]
        for i in range(1, n):
            transformed.append(A.ReplayCompose.replay(replay, image=clip[i])["image"])
        return np.array(transformed)

    def _get_ext(self, idx: int) -> str:
        """
        Method for getting the ext for a video clip
        Args:
            idx: the index of the sample
        Returns:
            str: the extention, since the filename does not contain it
        """
        dataset = self._video_datasets[idx]
        match dataset:
            case "le2i":
                return ".avi"
            case "GMDCSA24":
                return ".mp4"
            case "mcfd":
                return ".avi"
            case "OOPS":
                return ".mp4"
            case _:
                raise RuntimeError(f"Dataset not implemented yet ({dataset})")

    def _load_video(self, video_path: Path, time_steps: np.ndarray, dataset: str, corruption_threshold: int = 6) -> np.ndarray:
        """
        Function for loading videos, since the default torch codec doesn't work
        due to le2i videos having different fps, sizes, and omnifall dataset
        annotation lengths being different lengths -> this custom approach
        Args:
            video_path: the path to the video and filename
            time_steps: the start and endpoint for the clip
            dataset: the name of the dataset
            corruption_threshold: the threshold to not accept a clip (assertion error), the clip needs to be manually
                added to settings if you want to remove it
        Returns:
            np.ndarray: the loaded clip
        """
        assert time_steps[1] > time_steps[0]
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            if dataset == "le2i":
                stream.codec_context.skip_frame = "NONKEY"
            pts = float(stream.time_base)
            start_pts = int(time_steps[0] / pts) # pyav crashes if using np int
            end_pts = int(time_steps[1] / pts)
            container.seek(start_pts, stream=stream)
            capture_candidates = (np.linspace(time_steps[0], time_steps[1], self._video_len, endpoint=False) / pts).astype(np.int32)
            video = list()
            idx = 0
            for frame in container.decode(stream):
                frame_pts = frame.pts
                if frame_pts is None or frame_pts < capture_candidates[idx]:
                    continue
                idx += 1
                img = frame.to_ndarray(format="rgb24")
                h, w = img.shape[:-1]
                pad_y = max(0, (w - h) // 2)
                pad_x = max(0, (h - w) // 2)
                img = cv.copyMakeBorder(img, pad_y,  pad_y, pad_x, pad_x, cv.BORDER_CONSTANT)
                img = cv.resize(img, (self._settings.image_size, self._settings.image_size))
                video.append(img) 
                if frame_pts > end_pts or idx >= self._video_len:
                    break
        video = np.array(video)
        n = len(video)
        assert n >= corruption_threshold, f"Corrupt clip: {video_path}, {capture_candidates}, {start_pts}, {end_pts}, {n}"
        if n < self._video_len:
            pad = [video[-1]] * (self._video_len - len(video))
            return np.concatenate([video, pad], axis=0)
        return video


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")
