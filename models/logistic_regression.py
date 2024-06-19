import torch.nn as nn
import torch

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(30, 1)


    def forward(self, input):
        y_hat = torch.sigmoid(self.fc1(input))
        return y_hat

class MNISTLogisticRegression(nn.Module):
    def __init__(self):
        super(MNISTLogisticRegression, self).__init__()
        self.fc1 = nn.Linear(784, 1)


    def forward(self, input):
        y_hat = torch.sigmoid(self.fc1(input))
        return y_hat
