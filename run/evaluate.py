from torch.utils.data import DataLoader, Subset, Dataset
import torch
import torch.optim as optim
import torch.nn as nn
from subset.transformed_subset import TransformedSubset
from typing import Optional, Tuple
from transformations.standardize import Standardize

class Evaluate():
    def __init__(self,
        model: nn.Module,
        train_dataset,
        test_dataset,
        lr,
        criterion,
        optimizer,
        epochs,
        batch_size,
        val_dataset=None
        ):

        # Define parameters
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.lr = lr
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_dataset = val_dataset
        #self.device = device; Will add device (sending to GPU or TPU) functionality soon

        # Model selection using the valdation set
        self.best_loss = float('inf')
        self.best_model = None

        self.train()

    def train(self):
        self.model.train()

        # Check if user has a validation, if not make one
        if self.val_dataset:
            # Calculate standardization through the mean and std of train dataset
            mean, std = self.calc_mean_std(self.train_dataset)
            # Transformation for all the dataset
            transform = Standardize(mean=mean, std=std)
            self.train_dataset.transform = transform
            self.val_dataset.transform = transform
            self.test_dataset.transform = transform
        else:
            # Make validation set out of train set
            self.train_dataset, self.val_dataset = self.create_val_set()

        # PyTorch DataLoader class for more efficient data loading, shuffling the train data
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(self.val_dataset, batch_size=len(self.val_dataset), shuffle=False)
        test_dataloader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)

        # Trainig loop (Note, will add stopping criteria)
        for epoch in range(self.epochs):
            total_loss = 0.0
            self.model.train()
            for x, y in train_dataloader:
                self.optimizer.zero_grad()
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            epoch_loss = total_loss / len(train_dataloader)
            print(f"\nEpoch = {epoch}\nTraining loss = {epoch_loss}")

            # Calculate val loss for current epoch and check if it's better
            # then the last one, then set that model equal to the best model
            val_loss = self.val(val_dataloader=val_dataloader)
            if (val_loss < self.best_loss):
                self.best_loss = val_loss
                self.best_model = self.model.state_dict()

        # Load the best model that had the best val loss
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)

        # Accuracy on test set
        accuracy = self.test(test_dataloader=test_dataloader)

    # Validation algorithm (Note, will replace with KFold, using it for testing)
    def val(self, val_dataloader):
        self.model.eval()

        total_loss = 0.0

        # Go through the entire data set in DataLoader with a batch_size of the entire val set
        with torch.no_grad():
            for x, y in val_dataloader:
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)

                # Calculate the loss of the entire data set
                total_loss += loss.item()

        print(f"Validation loss = {total_loss}")

        return total_loss

    # Testing algorithm
    def test(self, test_dataloader):
        self.model.eval()
        correct = 0

        # Calculate the predictions by setting the tensor values equal to binary values
        with torch.no_grad():
            for x, y in test_dataloader:
                outputs = self.model(x)
                predictions = (outputs >= 0.5)

                # Sum how many correct values
                # predictions == y returns a tensor with 1's where they are equal, 0's otherwise
                correct += (predictions == y).sum()

        accuracy = correct / len(self.test_dataset)
        print(f"Accuracy = {accuracy}")
        return accuracy

    # Calculate the mean and std of the training data using DataLoader for efficiency
    def calc_mean_std(self, dataset):
        train_dataloader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset), shuffle=False)
        train_set = next(iter(train_dataloader))
        mean, std = train_set[0].mean(dim=0), train_set[0].std(dim=0)
        std[std == 0] = 1

        return mean, std

    # Create val set from train set if user doesn't have one
    def create_val_set(self):
        # Take 20% of train set for validation, shuffle the indices
        num_samples = len(self.train_dataset)
        num_val = int(0.2 * num_samples)
        shuffled_indices = torch.randperm(num_samples)

        # Get the indices for training and val sets
        train_indices = shuffled_indices[:-num_val].tolist()
        val_indices = shuffled_indices[-num_val:].tolist()

        # Create a subsetset of the actual train set
        train_subset = Subset(self.train_dataset, train_indices)
        val_subset = Subset(self.train_dataset, val_indices)

        # Calculate mean, std from train subset, then transform
        mean, std = self.calc_mean_std(train_subset)

        transform = Standardize(mean=mean, std=std)

        # Make a new Dataset class of each for faster loading and efficiency
        train_dataset = TransformedSubset(subset=train_subset, transform=transform)
        val_dataset = TransformedSubset(subset=val_subset, transform=transform)

        self.test_dataset.transform = transform

        return train_dataset, val_dataset
