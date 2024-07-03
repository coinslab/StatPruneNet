import torch.nn as nn
import torch

class SoftmaxRegression(nn.Module):
    def __init__(self, input_layer, output_layer):
        super(SoftmaxRegression, self).__init__()
        self.fc1 = nn.Linear(input_layer, 1)
        self.fc2 = nn.Linear(1, output_layer)

    def forward(self, input):
        input = input.view(-1, 28*28)
        y_hat = self.fc1(input)
        y_hat = self.fc2(y_hat)
        return y_hat
