from torch.utils.data import Subset, Dataset, DataLoader, ConcatDataset
import torch
from torch.optim import Optimizer
import torch.nn as nn
from typing import Optional, List
from torch.optim import Optimizer
from experiment.train import Train

class CrossValidation(Train):
    def __init__(self, k: Optional[int] = 10, seed: Optional[int] = None):
        self.k = k
        self.seed = seed

    def _kfolds(self, dataset: Dataset, len_dataset: int) -> List[Dataset]:
        if self.seed is not None:
            torch.manual_seed(self.seed)

        length = len_dataset
        indices = torch.randperm(length).tolist()

        fold_sizes = [length // self.k] * self.k
        for i in range(length % self.k):
            fold_sizes[i] += 1

        folds = []
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            folds.append(Subset(dataset, indices[start:stop]))
            current = stop

        return folds

    def validation(self,
                   model: nn.Module,
                   dataset: Dataset,
                   loss_fn: nn.Module,
                   optimizer: Optimizer,
                   device: torch.device,
                   epochs: Optional[int] = 20,
                   l1_approx_lambda: Optional[float] = 0.1) -> None:
        
        len_dataset = len(dataset)

        print("Starting KFold training")
        kfolds = self._kfolds(dataset, len_dataset)

        for fold in range(len(kfolds)):

            print(f"Fold = {fold + 1}")
            val_fold = kfolds[fold]

            train_folds = kfolds[:fold] + kfolds[fold + 1:]
            trainset = ConcatDataset(train_folds)

            val_loader = DataLoader(val_fold, batch_size=len(val_fold), shuffle=False, num_workers=2, pin_memory=True)
            trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=True, num_workers=2, pin_memory=True)

            x, y = next(iter(trainloader))
            x, y = x.to(device), y.to(device)
            loss = 0.0

            for epoch in range(epochs):
                def closure():
                    optimizer.zero_grad()
                    y_hat = model(x)
                    loss = loss_fn(y_hat, y) + self._log_cosh_regularization(l1_approx_lambda, len_dataset)
                    loss.backward()

                    return loss

                loss = optimizer.step(closure)

                print(f"\n\tEpoch = {epoch + 1}\tTraining loss = {loss:.2f}")
