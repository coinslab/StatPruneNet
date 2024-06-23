from torch.utils.data import Dataset
import torch
import numpy as np
import pathlib

class BinaryMNIST(Dataset):
    def __init__(self, root, train=True, transform=None):
        root = pathlib.Path(root)
        if train:
            data = np.loadtxt(root / 'binary_mnist/train.data', delimiter=',', dtype=np.float32)
        else: data = np.loadtxt(root / 'binary_mnist/test.data', delimiter=',', dtype=np.float32)

        self.X = torch.tensor(data[:, 1:], dtype=torch.float32)
        self.Y = torch.tensor(data[:, 0])
        self.len = self.X.size(0)
        self.transform= transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index].view(-1)

        if self.transform:
            x = self.transform(x)

        return x, y
