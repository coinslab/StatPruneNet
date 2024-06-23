from torch.utils.data import DataLoader
import torch.nn as nn

class Standardize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std
        self.std[std == 0] = 1

    def forward(self, tensor):
        return (tensor - self.mean) / self.std
