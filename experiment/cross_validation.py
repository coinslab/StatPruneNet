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

                '''
                
        print("Pruning model...")
        if self.model is not None:
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.random_unstructured(module, name='weight', amount=0.99)
                    prune.random_unstructured(module, name='bias', amount=0.99)


        print("Starting KFold training....")
        kfolds = KFold(self.train_dataset)
        k = len(kfolds)

        train_accuracies = torch.zeros(k, device=self.device)
        train_precisions = torch.zeros(k, device=self.device)
        train_recalls = torch.zeros(k, device=self.device)

        val_accuracies = torch.zeros(k, device=self.device)
        val_precisions = torch.zeros(k, device=self.device)
        val_recalls = torch.zeros(k, device=self.device)

        for fold in range(len(kfolds)):

            print(f"Fold = {fold + 1}")
            val_fold = kfolds[fold]

            train_folds = kfolds[:fold] + kfolds[fold + 1:]
            trainset = ConcatDataset(train_folds)

            # Using LBFGS optimizer, so the batch_size is the entire dataset (FashionMNIST)
            val_loader = DataLoader(val_fold, batch_size=len(val_fold), shuffle=False, num_workers=2)
            trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=True, num_workers=2)

            for epoch in range(self.epochs):
                for x, y in trainloader:
                    x, y = x.to(self.device), y.to(self.device)

                    def closure():
                        self.optimizer.zero_grad()
                        y_hat = self.model(x)
                        loss = self.criterion(y_hat, y)

                        # l2_lambda = 0.5 / (2 * len(self.train_dataset))
                        # l2_norm = sum(i.pow(2.0).sum() for i in self.model.parameters())
                        # loss += l2_lambda * l2_norm

                        # l1_lambda = 0.5 / len(self.train_dataset)
                        # l1_norm = sum(i.abs().sum() for i in self.model.parameters())
                        # loss += l1_lambda * l1_norm

                        loss.backward()

                        return loss

                    loss = self.optimizer.step(closure)

                print(f"\n\tEpoch = {epoch + 1}\tTraining loss = {loss.item():.2f}")

            train_accuracy, train_precision, train_recall = metrics(self.model, self.device, trainloader)
            train_accuracies[fold] = train_accuracy
            train_precisions[fold] = train_precision
            train_recalls[fold] = train_recall

            print(f"\n\tTraining Results\n\t----------------")
            print(f"\tAccuracy = {train_accuracy:.2f}%")
            print(f"\tPrecision = {train_precision:.2f}%")
            print(f"\trecall = {train_recall:.2f}%")

            val_accuracy, val_precision, val_recall = metrics(self.model, self.device, val_loader)
            val_accuracies[fold] = val_accuracy
            val_precisions[fold] = val_precision
            val_recalls[fold] = val_recall

            print(f"\n\tValidation Results\n\t----------------")
            print(f"\tAccuracy = {val_accuracy:.2f}%")
            print(f"\tPrecision = {val_precision:.2f}%")
            print(f"\trecall = {val_recall:.2f}%")

        train_accuracy = train_accuracies.mean()
        train_precision = train_precisions.mean()
        train_recall = train_recalls.mean()

        val_accuracy = val_accuracies.mean()
        val_precision = val_precisions.mean()
        val_recall = val_recalls.mean()

        print(f"Results\n----------------------------------------")
        print(f"Training Average\n----------------")
        print(f"Accuracy = {train_accuracy:.2f}%")
        print(f"Precision = {train_precision:.2f}%")
        print(f"recall = {train_recall:.2f}%")

        print(f"Validation Average\n----------------")
        print(f"Accuracy = {val_accuracy:.2f}%")
        print(f"Precision = {val_precision:.2f}%")
        print(f"recall = {val_recall:.2f}%")
        '''
