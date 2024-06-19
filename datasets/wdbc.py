from torch.utils.data import Dataset
import torch
import numpy as np
import pathlib
from sklearn.preprocessing import StandardScaler

class WDBC(Dataset):
    def __init__(self, root):
        root = pathlib.Path(root)
        data = np.loadtxt(root / 'wdbc/wdbc.data', delimiter=',', dtype=np.float32, converters={1: lambda x: 1 if x == 'M' else 0})
        X = data[:, 2:]
        sc = StandardScaler()
        X = sc.fit_transform(X)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(data[:, 1])
        self.len = self.X.size(0)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.X[index], self.Y[index].view(1)
