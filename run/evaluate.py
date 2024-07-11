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

    def compute_grad(self, x, y):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        prediction = self.model(x)
        loss = self.criterion(prediction, y)

        gradients = list(torch.autograd.grad(loss, list(self.model.parameters())))
        gradients = torch.cat([g.view(-1) for g in gradients])

        return gradients

    def compute_G(self, X, Y):
        G = torch.stack([self.compute_grad(X[i], Y[i]) for i in range(len(self.train_dataset))])

        return G

    def GRADMAX(self, G):
        average = G.mean(dim=0)
        absolute = average.abs()
        gradmax = absolute.max()

        return gradmax.item()

    def train(self):
        self.model.to(self.device)
        self.model.train()

        trainloader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=False, num_workers=0)

        for epoch in range(self.epochs):
            loss = 0.0
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)

                def closure():
                    self.optimizer.zero_grad()
                    y_hat = self.model(x)
                    loss = self.criterion(y_hat, y)

                    #params = list(self.model.parameters())
                    params = torch.cat([p.view(-1) for p in self.model.parameters()])

                    l2_lambda = 0.001 / (2 * len(self.train_dataset))
                    l2_norm = params.pow(2.0).sum() #sum(p.pow(2.0).sum() for p in self.model.parameters())
                    loss += l2_lambda * l2_norm

                    log_cosh_lambda = 0.001 / len(self.train_dataset)
                    log_cosh_norm = torch.log(torch.cosh(2.3099 * params)).sum() #sum(torch.log(torch.cosh(2.3099 * p)).sum() for p in self.model.parameters())
                    loss += log_cosh_lambda * log_cosh_norm

                    loss.backward()

                    return loss

                loss += self.optimizer.step(closure)

                G = self.compute_G(x, y)
                gradmax = self.GRADMAX(G)

                if (gradmax < 0.007):
                    print("GRADMAX value achieved")
                    return

            print(f"\n\tEpoch = {epoch + 1}\tTraining loss = {loss:.2f}")

        '''
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
