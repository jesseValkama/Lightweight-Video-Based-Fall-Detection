import argparse
from src.settings.settings import Settings


def handle_arguments(settings: Settings) -> None:
    """
    Handles command-line argumenets
    Args:
        settings: Settings object
    """
    parser = argparse.ArgumentParser(
        prog="Lightweight Video-Based Fall Detection Research Code",
        description="Trains and tests models for fall detection"
    )
    parser.add_argument("--train", default=0, choices=[0, 1], help="Whether to train the model: (0 | 1)", type=int)
    parser.add_argument("--test", default=0, choices=[0, 1], help="Whether to test the model: (0 | 1)", type=int)
    parser.add_argument("--inference", default=0, choices=[0, 1], help="Whether to use a model for inference: (0 | 1)", type=int)
    parser.add_argument("--cam", default="ScoreCAM", choices=["GradCAM, ScoreCAM"], help="Choose the visualisation CAM for inference: (ScoreCAM | GradCAM)", type=str)
    args = parser.parse_args()
    assert args.train + args.test + args.inference > 0, "Usage: python main.py"
    settings.train = args.train
    settings.test = args.test
    settings.inference = args.inference
    settings.inference_cam = args.cam


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")