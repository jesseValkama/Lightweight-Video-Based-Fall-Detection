import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from src.visualise.plot_container import PlotContainer
import torch
from torch import nn
from typing import List


class MetricsContainer:
    """
    Automates the metrics calculation for omnifall
        uses PlotContainer for tensorboard
    """

    def __init__(self, dataset_labels: List[str], plot_container: PlotContainer, activation_function = nn.Softmax) -> None:
        """
        initialises the object
        Args:
            dataset_labels: list of dataset_labels
            plot_container: wrapper for SummaryWriter
            activation_function: softmax (probably)
        """
        self._plot_container = plot_container
        self._dataset_labels = dataset_labels
        self._activation_function = activation_function(dim=1)
        n = len(dataset_labels)
        self._conf_mat_elements = {dataset_labels[i]: {"tp": 0, "fp": 0, "fn": 0} for i in range(n)}
        self._conf_mat_table = np.zeros((n, n))
        self._preds_total = np.zeros(n)
        self._preds_correct = np.zeros(n)
        self._10_class = None
        self._fall = None
        self._fallen = None 
        self._fall_U_fallen = None
        self._recall = None
        self._precision = None
        self._f1 = None
        self._embeddings = torch.Tensor([])
        self._labels = torch.Tensor([])
        
    def calc_iter(self, logits: torch.Tensor, labels: torch.Tensor, cls_weights: np.ndarray[float] | None = None) -> None:
        """
        used in the testing loop to construct the confusion matrix
        Args:
            logits: the output of the model before softmax
            labels: the corresponding labels for the outputs
            cls_weights: the cls_weights used for training (if used), removes the unwanted classes from metrics 
        """
        candidates = self._activation_function(logits)
        candidates, labels = candidates.detach().cpu().numpy(), labels.detach().cpu().numpy()
        preds = np.argmax(candidates, axis=1)
        
        n = len(preds)
        for i in range(n):
            if cls_weights is not None and cls_weights[labels[i]] == 0.0:
                print(f"skipped: {self._dataset_labels[labels[i]]}")
                continue
            self._conf_mat_table[labels[i]][preds[i]] += 1
            self._preds_total[labels[i]] += 1
            if preds[i] == labels[i]:
                self._conf_mat_elements[self._dataset_labels[preds[i]]]["tp"] += 1
                self._preds_correct[labels[i]] += 1
                continue
            self._conf_mat_elements[self._dataset_labels[preds[i]]]["fp"] += 1
            self._conf_mat_elements[self._dataset_labels[labels[i]]]["fn"] += 1

    def calc_metrics(self, cls_weights: np.ndarray[float] | None) -> None:
        """
        Calculates the metrics: sensitivity = recall, specificity = precision
        Args:
            cls_weights: the cls_weights used for training if used, otherwise None
        """
        if cls_weights is None:
            cls_weights = np.ones(len(self._dataset_labels), dtype=np.float32)
        classes = self._conf_mat_elements.keys()
        self._recall = np.array([ 
            self._conf_mat_elements[cls]["tp"]/(self._conf_mat_elements[cls]["tp"] + self._conf_mat_elements[cls]["fn"])
            if (self._conf_mat_elements[cls]["tp"] + self._conf_mat_elements[cls]["fn"] > 0) and (cls_weights[i] != 0.0) else self._metrics_helper(cls_weights, i)
            for i, cls in enumerate(classes) 
        ])
        self._precision = np.array([
            self._conf_mat_elements[cls]["tp"]/(self._conf_mat_elements[cls]["tp"] + self._conf_mat_elements[cls]["fp"]) 
            if (self._conf_mat_elements[cls]["tp"] + self._conf_mat_elements[cls]["fp"] > 0) and (cls_weights[i] != 0.0) else self._metrics_helper(cls_weights, i)
            for i, cls in enumerate(classes)
        ])
        self._f1 = np.array([
            2 * (self._precision[cls] * self._recall[cls]) / (self._precision[cls] + self._recall[cls])
            if (self._precision[cls] + self._recall[cls] > 0) and (cls_weights[cls] != 0.0) else self._metrics_helper(cls_weights, cls)
            for cls in range(len(classes)) 
        ])
        self._fall = {"recall": self._recall[1], "precision": self._precision[1], "f1": self._f1[1]}
        self._fallen = {"recall": self._recall[2], "precision": self._precision[2], "f1": self._f1[2]}
        self._fall_U_fallen = {"recall": np.mean(self._recall[1:3]), "precision": np.mean(self._precision[1:3]), "f1": np.mean(self._f1[1:3])}
        self._10_class = {"balanced_accuracy": (np.nanmean(self._precision) + np.nanmean(self._recall)) / 2, "accuracy": np.sum(self._preds_correct) / np.sum(self._preds_total), "f1": np.nanmean(self._f1)}

    def _metrics_helper(self, cls_weights: np.ndarray[float] | None, idx: int) -> float:
        """
        Helper method for the calc_metrics
        Args:
            cls_weights: you already know
            idx: for the class
        Returns:
            float: if the class is used, otherwise None
        """
        if cls_weights is None:
            return 0.0
        return np.nan if cls_weights[idx] == 0.0 else 0.0

    def add_embedding(self, embedding: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Method to add embeddings for embedding visualisations
        Args:
            embedding: the embeddings from the hook
            labels: the corresponding labels
        """
        self._embeddings = torch.cat((self._embeddings, embedding), dim=0)
        self._labels = torch.cat((self._labels, labels), dim=0)

    def tsne(self) -> None:
        """
        Method to visualise the embeddings in latent space with t-sne with tensorboard
        t-sne paper:
            https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
        """
        assert len(self._embeddings) != 0.0, "add embeddings with add_embedding method"
        tsne = TSNE()
        latent_repr = tsne.fit_transform(self._embeddings.numpy())
        self._plot_container.push_tsne(latent_repr, self._labels.numpy(), self._dataset_labels)
        
    def show_conf_mat(self) -> None:
        """
        Method to visualise the confusion matrix with tensorboard
        """
        self._plot_container.push_conf_mat(self._conf_mat_table, self._dataset_labels)

    def print_conf_mat(self) -> None:
        """
        Method to print the confusion matrix
        """
        assert np.sum(self._conf_mat_table) != 0.0, "Construct the confusion matrix first with calc_iter method"
        assert np.sum(self._conf_mat_table) == np.sum(self._preds_total), \
        f"conf mat messed up: conf mat {np.sum(self._conf_mat_table)}, total {np.sum(self._preds_total)}"
        print("---------------START OF THE CONFUSION MATRIX----------------\n\n")
        print(self._conf_mat_table)
        print("\n\n----------------END OF THE CONFUSION MATRIX-----------------")
        print("\n\n")
    
    def show_metrics(self, dec: int = 2) -> None:
        """
        Method to visualise the metrics with tensorboard
        Args:
            dec: the number of decimals
        """
        fall = pd.DataFrame(data={"sensitivity": [np.round(self._fall["recall"], dec), np.round(self._fallen["recall"], dec), np.round(self._fall_U_fallen["recall"], dec)], 
                                       "specificity": [np.round(self._fallen["precision"], dec), np.round(self._fallen["precision"], dec), np.round(self._fall_U_fallen["precision"], dec)],
                                       "f1": [np.round(self._fall["f1"], dec), np.round(self._fallen["f1"], dec), np.round(self._fall_U_fallen["f1"], dec)]}, index=["fall", "fallen", "fall U fallen"])
        ten_class = pd.DataFrame(data={"balanced accuracy": [np.round(self._10_class["balanced_accuracy"], dec)], "accuracy": [np.round(self._10_class["accuracy"], dec)], "f1": [np.round(self._10_class["f1"], dec)]},
                                 index=["10 class"])
        self._plot_container.push_test_metrics(fall, ten_class)

    def print_metrics(self) -> None:
        """
        Method to print the metrics
        """
        assert self._10_class is not None, "Calculate the metrics before printing"
        n = len(self._dataset_labels)
        print("-------------------START OF THE METRICS-------------------\n\n")

        print("---------------------------ALL----------------------------\n\n")
        for i in range(n):
            print(f"{self._dataset_labels[i]}:")
            print(f"Recall: {self._recall[i]:.2f}")
            print(f"Precision: {self._precision[i]:.2f}")
            print(f"F1: {self._f1[i]:.2f}\n")
        
        print("\n-------------------------OMNIFALL-------------------------\n\n")
        print("10-class")
        print(f"Balanced accuracy: {self._10_class["balanced_accuracy"]}")
        print(f"Accuracy: {self._10_class["accuracy"]}")
        print(f"F1: {self._10_class["f1"]}\n")

        print("Fall")
        print(f"Recall: {self._fall["recall"]}")
        print(f"Precision: {self._fall["precision"]}")
        print(f"F1: {self._fall["f1"]}\n")

        print("Fallen")
        print(f"Recall: {self._fallen["recall"]}")
        print(f"Precision: {self._fallen["precision"]}")
        print(f"F1: {self._fallen["f1"]}\n")

        print("Fall U Fallen")
        print(f"Recall: {self._fall_U_fallen["recall"]}")
        print(f"Precision: {self._fall_U_fallen["precision"]}")
        print(f"F1: {self._fall_U_fallen["f1"]}\n")

        print("\n--------------------END OF THE METRICS--------------------")
        print("\n\n")


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")