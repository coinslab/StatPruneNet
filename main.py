from models.logistic_regression import MNISTLogisticRegression
from datasets.binary_mnist import BinaryMNIST
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch
import torch.nn as nn
from run.evaluate import Evaluate


# Params
model = MNISTLogisticRegression()
train_dataset = BinaryMNIST(root='./data', train=True)
test_dataset = BinaryMNIST(root='./data', train=False)
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
epochs = 1
batch_size = 64

# Training, validation, and testing
Evaluate(
    model=model,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    lr=learning_rate,
    criterion=criterion,
    optimizer=optimizer,
    epochs=epochs,
    batch_size=batch_size)
