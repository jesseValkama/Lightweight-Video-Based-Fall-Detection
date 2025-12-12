import os
from pathlib import Path
from src.inference.inference import run_inference
from src.train.train_loop import run_loop 
from src.settings.handle_args import handle_arguments
from src.settings.settings import Settings


def main() -> None:
    """
    This is the main file, run everything from here
    Uses both command-line arguments and also src/settings/settings.py
    Command-line args:
        train: 0 | 1
        test: 0 | 1
        inference: 0 | 1
        cam: ScoreCAM | GradCAM
    """
    settings = Settings()
    handle_arguments(settings)
    project_dir = Path(settings.project_dir)
    assert project_dir.exists, "Enter a valid project directory, no need to include main.py"
    os.chdir(project_dir)
    if settings.train or settings.test:
        run_loop(settings=settings)
    if settings.inference:
        run_inference(settings, settings.inference_cam)


if __name__ == "__main__":
    main()