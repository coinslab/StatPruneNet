from torch.utils.data import Subset
import torch

def KFold(dataset, k=10, random_state=None):
    if random_state is not None:
        torch.manual_seed(random_state)
    length = len(dataset)
    indices = torch.randperm(length).tolist()

    fold_sizes = [length // k] * k
    for i in range(length % k):
        fold_sizes[i] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(Subset(dataset, indices[start:stop]))
        current = stop

    return folds
