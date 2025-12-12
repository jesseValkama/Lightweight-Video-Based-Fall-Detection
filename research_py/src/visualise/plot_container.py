import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.settings.settings import Settings
from torch.utils.tensorboard import SummaryWriter
from typing import List


class PlotContainer:
    """
    Wrapper for the torch SummaryWriter
    Automatically plots graphs for runtime
    and metrics from the testing
    Recommended:
        Run the tensorboard from a separate terminal with:
        tensorboard --logdir=runs
    """

    def __init__(self, writer: SummaryWriter, settings: Settings) -> None:
        """
        Args:
            writer: the created summary writer
            settings: the settings object
        """
        self._writer = writer
        self._settings = settings
        self._train_data = {
            "train": {
                "loss": list(),
                "accuracy": list() 
            },
            "val": {
                "loss": list(),
                "accuracy": list() 
            }
        }

    def update_train_plots(self, loss: float, accuracy: float, type: str) -> None:
        """
        updates the tensorboard live plots
        Args:
            loss: the loss from the current epoch
            accuracy: the loss from the current epoch
            type: train or val
        """
        if not type in self._train_data:
            raise RuntimeError(f"Incorrect type for updating a plot: {type}")
        self._train_data[type]["loss"].append(loss)
        self._train_data[type]["accuracy"].append(accuracy)
        epoch = len(self._train_data["train"]["loss"])
        self._writer.add_scalar(type + " loss", loss, global_step=epoch)
        self._writer.add_scalar(type + " accuracy", accuracy, global_step=epoch)
                
    def push_train_plots(self) -> None:
        """
        pushes the training and validation loss to the same plot to tensorboard
        """
        epoch = len(self._train_data["train"]["loss"])
        assert epoch > 0, "you need to update the plots with update_plot before pushing"
        fig, axs = plt.subplots(2, 1)
        fig.suptitle("Training metrics")
        plt.subplots_adjust(hspace=0.5)
        x = np.linspace(1, epoch, epoch)
        x_val = np.arange(self._settings.validation_interval, epoch + 1, self._settings.validation_interval)
        train_loss = np.array(self._train_data["train"]["loss"])
        val_loss = np.array(self._train_data["val"]["loss"])
        train_accuracy = np.array(self._train_data["train"]["accuracy"])
        val_accuracy = np.array(self._train_data["val"]["accuracy"])
        val_loss_interp = np.interp(x, x_val, val_loss)
        val_accuracy_interp = np.interp(x, x_val, val_accuracy)

        axs[0].plot(x, train_loss, c="g", label="Train loss")
        axs[0].plot(x, val_loss_interp, c="b", label="Validation loss")
        axs[0].scatter(x, train_loss, c="g")
        axs[0].scatter(x_val, val_loss, c="b")
        axs[0].set_ylabel("Loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_title("Loss curves")
        axs[0].legend()
        axs[1].plot(x, train_accuracy, c="g", label="Train accuracy")
        axs[1].plot(x, val_accuracy_interp, c="b", label="Validation accuracy")
        axs[1].scatter(x, train_accuracy, c="g")
        axs[1].scatter(x_val, val_accuracy, c="b")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_xlabel("Epoch")
        axs[1].set_title("Accuracy curves")
        axs[1].legend()
        self._writer.add_figure("Training metrics", fig)
    
    def push_conf_mat(self, cm: np.ndarray, labels: List[str]) -> None:
        """
        pushes the confusion matrix to tensorboard
        Acknowledgements:
            https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
        Args:
            cm: the confusion matrix as np.array
            labels: the dataset labels as a list
        """
        fig = plt.figure()
        df_cm = pd.DataFrame(cm, index=[l for l in labels], columns=[l for l in labels])
        sns.heatmap(df_cm, annot=True)
        self._writer.add_figure("Confusion Matrix", fig)
    
    def push_test_metrics(self, ten_class: pd.DataFrame, fall: pd.DataFrame) -> None:
        """
        pushes the testing metrics to tensorboard
        Args:
            ten_class: the metrics for ten class
            fall: the metrics related to fall and fallen classes
        """
        fig, axs = plt.subplots(2, 1)
        axs[0].axis("off")
        axs[0].table(cellText=ten_class, loc="center", cellLoc="center")
        axs[1].axis("off")
        axs[1].table(cellText=fall, loc="center", cellLoc="center")
        plt.tight_layout()
        self._writer.add_figure("Test metrics", fig)
    
    def push_tsne(self, latent_repr: np.ndarray, labels: np.ndarray, dataset_labels: List[str]) -> None:
        """
        pushes the t-sne visualisation to tensorboard
        Acknowledgements:
            https://builtin.com/data-science/tsne-python
        Args:
            latent_repr: the embeddings visualised in the latent space
            labels: the corresponding labels for the embeddings
            dataset_labels: the dataset labels as a list
        """
        fig = plt.figure()
        df = pd.DataFrame()
        df["x"] = latent_repr[:, 0]
        df["y"] = latent_repr[:, 1]
        df["labels"] = labels
        df["labels"] = df["labels"].map({float(i): label for i, label in enumerate(dataset_labels)})
        ax = sns.scatterplot(
            x = "x",
            y = "y",
            hue="labels",
            palette=sns.color_palette("hls", len(dataset_labels)),
            data=df,
            legend="full",
            alpha=0.6
        )
        ax.legend(loc="upper left", bbox_to_anchor=(-0.5, 0.8))
        plt.tight_layout()
        self._writer.add_figure("Embeddings in the latent space", fig)
    

if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")
    