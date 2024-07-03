from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Optional, Tuple, Type
from .kfold import KFold
from metrics import metrics

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

        for fold in range(len(kfolds)):
            self.model.load_state_dict(self.default_model)
            self.optimizer.load_state_dict(self.default_optimizer)

            print(f"Fold = {fold + 1}")
            val_fold = kfolds[fold]

            train_folds = kfolds[:fold] + kfolds[fold + 1:]
            trainset = ConcatDataset(train_folds)

            val_loader = DataLoader(val_fold, batch_size=len(val_fold), shuffle=False)
            trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=True)

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

            accuracy = metrics.accuracy(self.model, self.device, val_loader)
            precision = metrics.precision(self.model, self.device, val_loader)
            recall = metrics.recall(self.model, self.device, val_loader)

    def test(self, val_loader):
        self.model.eval()
        correct = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)

                _, predictions = torch.max(outputs, 1)

                correct += (predictions == y).sum().item()

        accuracy = correct / len(val_loader.dataset) * 100
        print(f"\tAccuracy = {accuracy:.2f}\n")
        return accuracy
