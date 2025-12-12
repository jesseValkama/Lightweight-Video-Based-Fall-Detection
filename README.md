# Light-weight Video Based Fall Detection

This is my bachelor's final project. 
Paper is available here (COMING SOON I GUESS)

## Description

The project is a study (proposed by school) for fall detection with lightweight models (CNNs + RNNs). The Omnifall benchmark is used for evalution.
This project contains the training loop, the models, testing and inference. However, model weights are not released. There is support for tensorboard and support for inference.

Results:

## Getting Started

### Dependencies

* CUDA 12.6

### Executing program

You might need to change some dependencies in the src/settings/settings.py file
Also all of the training settings should be defined there
```
python main.py args
```
```
tensorboard --logdir=runs
```
args:

    train: 0 | 1

    test: 0 | 1

    inference: 0 | 1

    cam: "ScoreCAM" | "GradCAM"

## Authors

Jesse Valkama

## Acknowledgments

MODELS
* [EFFICIENTNET](https://arxiv.org/pdf/1905.11946)
* [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)
* [LRCN](https://arxiv.org/abs/1411.4389)

DATASETS (omnifall annotations for le2i, GMDCSA24, OOPS)
* [OMNIFALL](https://arxiv.org/abs/2505.19889)
* [LE2I](https://search-data.ubfc.fr/imvia/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html)
* [GMDCSA24](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos)
* [OOPS](https://oops.cs.columbia.edu/data/)

OTHER
* [README TEMPLATE](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)

