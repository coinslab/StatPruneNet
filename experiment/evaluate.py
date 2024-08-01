from torch.utils.data import DataLoader, ConcatDataset
import torch
import torch.optim as optim
import torch.nn as nn
from typing import Optional, Tuple, Type
from .kfold import KFold
from metrics.metrics import metrics
from torch.func import functional_call, grad, vmap
import torch.nn.functional as F
import torch.nn.utils.prune as prune


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

        self.trained_model = None
        self.trained_optimizer = None

        self.train()
    '''
    # Vectorized implementation of G and GRADMAX
    def compute_G_vectorized(self, x, y):
        def compute_loss(params, x, y):
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

            y_hat = functional_call(self.model, params, x)

            loss = F.cross_entropy(y_hat, y)
            return loss

        params = {k: v.detach() for k, v in self.model.named_parameters()}

        grad_loss = grad(compute_loss)
        all_grads = vmap(grad_loss, in_dims=(None, 0, 0))
        grads = all_grads(params, x, y)
        grads = list(grads.values())
        G = torch.cat([g.view(len(self.train_dataset), -1) for g in grads], dim=1)

        gradmax = G.mean(dim=0).abs().max().item()

        return G, gradmax

    def train(self):
        self.model.to(self.device)
        self.model.train()

        trainloader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=False, num_workers=0)

        for epoch in range(self.epochs):
            gradmax = float('inf')
            converged = False
            loss = 0.0
            G = None
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)

                def closure():
                    self.optimizer.zero_grad()
                    y_hat = self.model(x)
                    loss = self.criterion(y_hat, y)

                    params = torch.cat([p.view(-1) for p in self.model.parameters()])

                    l2_lambda = 0.1 / (2 * len(self.train_dataset))
                    l2_norm = params.pow(2.0).sum()
                    loss += l2_lambda * l2_norm

                    log_cosh_lambda = 0.1 / len(self.train_dataset)
                    log_cosh_norm = torch.log(torch.cosh(2.3099 * params)).sum()
                    loss += log_cosh_lambda * log_cosh_norm

                    loss.backward()

                    return loss

                loss += self.optimizer.step(closure)

                G, gradmax = self.compute_G_vectorized(x, y)

            print(f"\n\tEpoch = {epoch + 1}\tTraining loss = {loss:.4f}\tGradmax = {gradmax:.4f}")

            if (gradmax < 0.0007):
                converged = True

            if converged:
                B = torch.matmul(G.T, G)
                self.trained_model = self.model.state_dict()
                self.trained_optimizer = self.optimizer.state_dict()
                print("GRADMAX value achieved!")
                print("Training complete!")
                break
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
