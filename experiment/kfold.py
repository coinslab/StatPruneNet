from torch.utils.data import Dataset, Subset, ConcatDataset
import torch
import torch.nn as nn
import copy
from .train import train
from .test import test
from utils.print_metrics import print_metrics
from utils.mean_std import compute_mean_std_err

def kfolds(dataset: Dataset, k: int = 10) -> list[Dataset]:
    length = len(dataset)
    indices = torch.randperm(length).tolist()

    fold_sizes = [length // k] * k
    for i in range(length % k):
        fold_sizes[i] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(Subset(dataset, indices[start:stop]))
        current = stop

    return folds

def kfold(model: nn.Module,
          dataset: Dataset,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          gmin: float,
          l2_lambda: float,
          l1_approx_lambda: float,):

    folds = kfolds(dataset)

    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())

    train_loss_list = []
    train_accuracy_list = []
    train_precision_list = []
    train_recall_list = []
    train_f1_list = []

    val_loss_list = []
    val_accuracy_list = []
    val_precision_list = []
    val_recall_list = []
    val_f1_list = []

    for fold in range(len(folds)):
        print(f"Fold = {fold + 1}")

        val_fold = folds[fold]

        train_folds = folds[:fold] + folds[fold + 1:]
        train_fold = ConcatDataset(train_folds)

        trained_model = train(model=model,
                              train_dataset=train_fold,
                              criterion=criterion,
                              optimizer=optimizer,
                              epochs=epochs,
                              device=device,
                              gmin=gmin,
                              l2_lambda=l2_lambda,
                              l1_approx_lambda=l1_approx_lambda,
                              train_only=False)
        
        trained_metrics = test(model=trained_model, test_dataset=train_fold, device=device)
        print('In-Sample Metrics')
        print_metrics(trained_metrics)

        train_loss_list.append(trained_metrics['loss'])
        train_accuracy_list.append(trained_metrics['accuracy'])
        train_precision_list.append(trained_metrics['precision'])
        train_recall_list.append(trained_metrics['recall'])
        train_f1_list.append(trained_metrics['f1'])

        val_metrics = test(model=trained_model, test_dataset=val_fold, device=device)
        print('Out-Sample metrics')
        print_metrics(val_metrics)

        val_loss_list.append(val_metrics['loss'])
        val_accuracy_list.append(val_metrics['accuracy'])
        val_precision_list.append(val_metrics['precision'])
        val_recall_list.append(val_metrics['recall'])
        val_f1_list.append(val_metrics['f1'])

        model.load_state_dict(initial_model_state)
        optimizer.load_state_dict(initial_optimizer_state)

    print('Final Metrics Across All Folds')
    train_loss_mean, train_loss_std_err = compute_mean_std_err(train_loss_list)
    train_accuracy_mean, train_accuracy_std_err = compute_mean_std_err(train_accuracy_list)
    train_precision_mean, train_precision_std_err = compute_mean_std_err(train_precision_list)
    train_recall_mean, train_recall_std_err = compute_mean_std_err(train_recall_list)
    train_f1_mean, train_f1_std_err = compute_mean_std_err(train_f1_list)

    print("Training Metrics")
    print(f"Loss: {train_loss_mean:.4f} +/- {train_loss_std_err:.4f}")
    print(f"Accuracy: {train_accuracy_mean * 100:.4f}% +/- {train_accuracy_std_err:.4f}")
    print(f"Precision: {train_precision_mean * 100:.4f}% +/- {train_precision_std_err:.4f}")
    print(f"Recall: {train_recall_mean * 100:.4f}% +/- {train_recall_std_err:.4f}")
    print(f"F1: {train_f1_mean * 100:.4f}% +/- {train_f1_std_err:.4f}")

    val_loss_mean, val_loss_std_err = compute_mean_std_err(val_loss_list)
    val_accuracy_mean, val_accuracy_std_err = compute_mean_std_err(val_accuracy_list)
    val_precision_mean, val_precision_std_err = compute_mean_std_err(val_precision_list)
    val_recall_mean, val_recall_std_err = compute_mean_std_err(val_recall_list)
    val_f1_mean, val_f1_std_err = compute_mean_std_err(val_f1_list)

    print("Validation Metrics")
    print(f"Loss: {val_loss_mean:.4f} +/- {val_loss_std_err:.4f}")
    print(f"Accuracy: {val_accuracy_mean * 100:.4f}% +/- {val_accuracy_std_err:.4f}")
    print(f"Precision: {val_precision_mean * 100:.4f}% +/- {val_precision_std_err:.4f}")
    print(f"Recall: {val_recall_mean * 100:.4f}% +/- {val_recall_std_err:.4f}")
    print(f"F1: {val_f1_mean * 100:.4f}% +/- {val_f1_std_err:.4f}")

    return