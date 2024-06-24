import torch.nn as nn
import torch

class LogisticRegression(nn.Module):
    def __init__(self, input_layer, output_layer):
        super(LogisticRegression, self).__init__()
        self.fc1 = nn.Linear(input_layer, output_layer)


    def forward(self, input):
        y_hat = self.fc1(input)
        return y_hat
