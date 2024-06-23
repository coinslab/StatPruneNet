from torch.utils.data import Dataset
import torch
import numpy as np
import pathlib

class WDBC(Dataset):
    def __init__(self, root, transform=None):
        root = pathlib.Path(root)
        data = np.loadtxt(root / 'wdbc/wdbc.data', delimiter=',', dtype=np.float32, converters={1: lambda x: 1 if x == 'M' else 0})
        self.X = torch.tensor(data[:, 2:], dtype=torch.float32)
        self.Y = torch.tensor(data[:, 1])
        self.len = self.X.size(0)
        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x, y =  self.X[index], self.Y[index].view(1)

        if self.transform:
            x = self.transform(x)

        return x, y
