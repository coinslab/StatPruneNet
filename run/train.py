from torch.utils.data import DataLoader, Subset
import torch
import torch.nn as nn
from .val import Validation

class Training():
    def __init__(self, model, dataset, lr, criterion, optimizer, epochs, batch_size):
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.best_loss = float('inf')
        self.best_model = None

        self.train()

    def train(self):
        self.model.train()

        num_samples = len(self.dataset)
        num_val = int(0.2 * num_samples)
        shuffled_indices = torch.randperm(num_samples)

        train_indices = shuffled_indices[:-num_val].tolist()
        val_indices = shuffled_indices[-num_val:].tolist()

        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)


        total_loss = 0.0
        total_samples = 0
        for epoch in range(self.epochs):
            for input, label in train_dataloader:
                self.optimizer.zero_grad()
                y_hat = self.model(input)

                loss = self.criterion(y_hat, label)
                batch_size = label.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                loss.backward()

                self.optimizer.step()


               #print(f"Loss = {loss.item()}")

            avg_loss = total_loss / total_samples

            print(f"\nEpoch = {epoch}\nTraining average loss = {avg_loss}")

            self.validation(val_dataset)


    def validation(self, val_dataset):
        val_loss = Validation(model=self.model, dataset=val_dataset, criterion=self.criterion, batch_size=len(val_dataset)).avg_loss
        if (val_loss < self.best_loss):
            self.best_loss = val_loss
            self.best_model = self.model.state_dict()

        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
