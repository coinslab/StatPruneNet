from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import pathlib

class BinaryMNIST(Dataset):
    def __init__(self, root):
        root = pathlib.Path(root)
        data = np.loadtxt(root / 'binary_mnist/train.data', delimiter=',', dtype=np.float32)
        X = data[:, 1:]
        sc = StandardScaler()
        X = sc.fit_transform(X)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(data[:, 0])
        self.len = self.X.size(0)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.X[index], self.Y[index].view(-1)
