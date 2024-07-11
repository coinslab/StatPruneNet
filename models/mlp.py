import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_layer, output_layer):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_layer, 10)
        self.fc2 = nn.Linear(10, output_layer)
        self.activation = nn.Softplus()

    def forward(self, input):
        input = input.view(-1, 28*28)
        y_hat = self.activation(self.fc1(input))
        y_hat = self.fc2(y_hat)
        return y_hat
