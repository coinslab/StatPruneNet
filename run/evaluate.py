from torch.utils.data import DataLoader, ConcatDataset
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Optional, Tuple, Type
from .kfold import KFold
from metrics.metrics import metrics

class Evaluate():
    def __init__(self,
        model: nn.Module,
        train_dataset,
        test_dataset,
        criterion,
        optimizer,
        epochs: int,
        ):

        # Define parameters
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.default_model = self.model.state_dict()
        self.default_optimizer = self.optimizer.state_dict()

        self.train()

    def train(self):
        self.model.to(self.device)

        kfolds = KFold(self.train_dataset)
        k = len(kfolds)

        train_accuracies = torch.zeros(k, device=self.device)
        train_precisions = torch.zeros(k, device=self.device)
        train_recalls = torch.zeros(k, device=self.device)

        val_accuracies = torch.zeros(k, device=self.device)
        val_precisions = torch.zeros(k, device=self.device)
        val_recalls = torch.zeros(k, device=self.device)

        for fold in range(len(kfolds)):
            self.model.load_state_dict(self.default_model)
            self.optimizer.load_state_dict(self.default_optimizer)

            print(f"Fold = {fold + 1}")
            val_fold = kfolds[fold]

            train_folds = kfolds[:fold] + kfolds[fold + 1:]
            trainset = ConcatDataset(train_folds)

            val_loader = DataLoader(val_fold, batch_size=len(val_fold), shuffle=False, num_workers=2)
            trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=True, num_workers=2)

            for epoch in range(self.epochs):
                self.model.train()
                for x, y in trainloader:
                    x, y = x.to(self.device), y.to(self.device)

                    def closure():
                        self.optimizer.zero_grad()
                        y_hat = self.model(x)
                        loss = self.criterion(y_hat, y)
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
