import torch.nn as nn

class SoftmaxMLP(nn.Module):
    def __init__(self, input_layer, output_layer):
        super().__init__()
        self.fc1 = nn.Linear(input_layer, 2)
        self.fc2 = nn.Linear(2, output_layer)
        self.activation = nn.Softplus()

    def forward(self, input):
        input = input.view(input.shape[0], -1)
        y_hat = self.activation(self.fc1(input))
        y_hat = self.fc2(y_hat)
        return y_hat
