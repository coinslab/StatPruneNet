from torch.utils.data import DataLoader
import torch
import torch.nn as nn

class Training:
    def __init__(self, model, dataset, lr, criterion, optimizer, epochs, batch_size):
        self.model = model
        self.dataset = dataset
        self.lr = lr
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

        self.train()

    def train(self):
        dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for input, label in dataloader:
                self.optimizer.zero_grad()
                y_hat = self.model(input)

                loss = self.criterion(y_hat, label)

                loss.backward()

                self.optimizer.step()

                print(f"Loss = {loss.item()}")
