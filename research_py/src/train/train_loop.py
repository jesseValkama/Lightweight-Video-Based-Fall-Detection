import math
import numpy as np
import os
from src.datasets.get_data_loader import get_data_loader
from src.datasets.load_omnifall import load_omnifall_info
from src.datasets.load_ucf101_info import load_ucf101_info
from src.datasets.omnifall import get_omnifall_datasets
from src.loss.get_criterion import get_criterion
from src.loss.trades import TRADES
from src.metrics.metrics_container import MetricsContainer
from src.models.efficientnet_lrcn import EfficientLRCN
from src.settings.settings import Settings
from src.utils.balance import get_cls_weights 
from src.utils.early_stop import EarlyStop
from src.utils.warmup import WarmupLR
from src.visualise.plot_container import PlotContainer
import time
import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def run_loop(settings: Settings) -> None:
    """
    main training loop fn (currently no support for ucf101 as it was never finished)
    Args:
        settings: settings for the training
    """
    writer = SummaryWriter()
    plot_container = PlotContainer(writer, settings)
    info_fn = load_omnifall_info if settings.dataset == "omnifall" else load_ucf101_info
    ds_info = info_fn(settings)
    samples = ds_info["train"]["samples"]
    labels = ds_info["train"]["labels"]
    train_set, val_set, test_set = get_omnifall_datasets(ds_info, settings)
    if settings.train:
        train_loader = get_data_loader(train_set, settings.train_batch_size, True, settings.train_num_workers, settings.async_transfers, drop_last=True)
        val_loader = get_data_loader(val_set, settings.val_batch_size, False, settings.val_num_workers, settings.async_transfers)
        train(train_loader, val_loader, samples, labels, plot_container, settings)
    if settings.test:
        test_loader = get_data_loader(test_set, settings.test_batch_size, False, settings.test_num_workers, settings.async_transfers)
        test(test_loader, samples, plot_container, settings)
       

def train(train_loader: DataLoader, val_loader: DataLoader, samples: np.ndarray, labels: np.ndarray, plot_container: PlotContainer, settings: Settings) -> None:
    """
    Training fn for stochastic gradient descent, includes trades if enabled (untested), potential for self-adaptive training
    Args:
        train_loader: the DataLoader for training
        val_loader: the DataLoader for validation
        samples: the number of samples per class
        labels: the labels for the samples
        plot_container: wrapper for the SummaryWriter
        settings: settings for training
    """
    model = EfficientLRCN(settings)
    model = model.to(settings.train_dev)
    cls_weights = get_cls_weights(samples, settings) if settings.apply_cls_weights else None 
    criterion = get_criterion(settings.criterion, settings, labels, cls_weights)
    trades = TRADES(settings)
    optimiser = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, settings.max_epochs)
    warmup = WarmupLR(optimiser, settings.warmup_length, settings.warmup_length * math.ceil(np.sum(samples) / settings.train_batch_size))
    scaler = GradScaler()
    training_loss = 0.0
    val_loss = 0.0
    best_val_loss = np.inf
    early_stop = EarlyStop(settings.min_epochs, settings.early_stop_tries)
    start_time = time.time()

    for epoch in range(1, settings.max_epochs+1):
        print(f"Starting epoch: {epoch}")
        training_loss = 0.0
        train_correct = 0.0
        train_total = 0.0
        model.train()

        for i, (vids, labels, label_idxs) in enumerate(train_loader):
            vids, labels = vids.to(settings.train_dev, non_blocking=settings.async_transfers), labels.to(settings.train_dev, non_blocking=settings.async_transfers).view(-1)
            optimiser.zero_grad()
            with autocast(device_type="cuda"):
                if settings.trades:
                    adversarial_examples = trades(model, vids)
                    optimiser.zero_grad()
                outputs = model(vids)
                loss = criterion(outputs, labels, reduction=True, apply_cls_weights=settings.apply_cls_weights, label_idxs=label_idxs, epoch=epoch)
                if settings.trades:
                    loss = trades.calc_loss(loss, adversarial_examples)
            iter_loss = loss.item()
            assert not math.isnan(iter_loss), "Training is unstable, change settings"
            training_loss += iter_loss
            idxs = torch.argmax(outputs, dim=1)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            for idx in range(len(idxs)):
                train_total += 1.0
                if idxs[idx] == labels[idx]:
                    train_correct += 1.0
            if epoch <= settings.warmup_length:
                warmup.step()

        training_loss /= (i + 1)
        training_accuracy = train_correct / train_total
        plot_container.update_train_plots(training_loss, training_accuracy, "train")
        if epoch % settings.validation_interval == 0 and epoch != 0:
            val_loss = validate(model, val_loader, plot_container, criterion, settings)
            improvement = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improvement = True
                print("The model improved")
                save_path = os.path.join(settings.weights_path, settings.work_model + ".pth")
                torch.save(model.state_dict(), save_path)
            if early_stop(epoch, improvement):
                break
        cosine_annealing.step()

    plot_container.push_train_plots()
    end_time = time.time()
    training_time = end_time - start_time
    # credit: https://www.geeksforgeeks.org/python/python-program-to-convert-seconds-into-hours-minutes-and-seconds/
    print("\n\n----------------------The training times----------------------\n\n")
    print("Total: ", time.strftime("%H:%M:%S", time.gmtime(training_time)))
    print("Per epoch: ", time.strftime("%H:%M:%S", time.gmtime(training_time / epoch)))
    print("\n\n---------------------End of training times---------------------\n\n")


@torch.no_grad
def validate(model: EfficientLRCN, val_loader: DataLoader, plot_container: PlotContainer, criterion: nn.CrossEntropyLoss, settings: Settings) -> float:
    """
    Validation fn
    Args:
        val_loader: DataLoader for validation
        plot_container: Wrapper for SummaryWriter
        criterion: the criterion used for training, if SAT is used, it is manually overwritten with epoch = 0
        settings: the settings for validation
    Return:
        float: validation loss
    """
    model.eval()
    validation_loss = 0.0
    val_total = 0.0
    val_correct = 0.0

    for i, (vids, labels, label_idxs) in enumerate(val_loader):
        vids, labels = vids.to(settings.train_dev, non_blocking=settings.async_transfers), labels.to(settings.train_dev, non_blocking=settings.async_transfers).view(-1)
        with autocast(device_type="cuda"):
            outputs = model(vids)
            loss = criterion(outputs, labels, reduction=True, apply_cls_weights=settings.apply_cls_weights, label_idxs=label_idxs, epoch=0) # epoch = 0 means that even if sat = True, it will use the actual criterion
        idxs = torch.argmax(outputs, dim=1)
        iter_loss = loss.item()
        assert not math.isnan(iter_loss), "validation is unstable, change settings"
        validation_loss += iter_loss
        for idx in range(len(idxs)):
            val_total += 1.0
            if idxs[idx] == labels[idx]:
                val_correct += 1.0

    validation_loss /= (i + 1)
    validation_accuracy = val_correct / val_total
    plot_container.update_train_plots(validation_loss, validation_accuracy, "val")
    return validation_loss


@torch.no_grad
def test(test_loader: DataLoader, samples: np.ndarray, plot_container: PlotContainer, settings: Settings) -> None:
    """
    Testing fn
    Args:
        test_loader: DataLoader for testing
        samples: the number of samples per class
        plot_container: wrapper for SummaryWriter
        settings: settings for testing
    """
    print("Starting testing")
    model_name = settings.work_model if settings.train else settings.test_model
    save_path = os.path.join(settings.weights_path, model_name + ".pth")
    model = EfficientLRCN(settings)
    model.load_state_dict(torch.load(save_path))
    model.to(settings.train_dev)
    model.eval()
    metrics_container = MetricsContainer(settings.dataset_labels, plot_container)
    hook_handle = model.rnn.register_forward_hook(
        lambda module, input, output : metrics_container.add_embedding(output[0][:,-1,:].cpu(), labels.cpu())
    )
    cls_weights = get_cls_weights(samples, settings, print_weights=False) if settings.apply_cls_weights else None

    for (vids, labels, _) in test_loader:
        vids, labels = vids.to(settings.train_dev, non_blocking=settings.async_transfers), labels.to(settings.train_dev, non_blocking=settings.async_transfers).view(-1)
        with torch.autocast(device_type="cuda"):
            outputs = model(vids)
        metrics_container.calc_iter(outputs, labels, cls_weights=cls_weights)
    hook_handle.remove()

    print("\n\n")
    metrics_container.tsne()
    metrics_container.show_conf_mat()
    metrics_container.print_conf_mat() 
    metrics_container.calc_metrics(cls_weights)
    metrics_container.show_metrics()
    metrics_container.print_metrics()


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")