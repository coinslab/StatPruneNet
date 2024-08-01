import torch.nn.utils.prune as prune
import torch.nn as nn
import torch

class Prune(prune.BasePruningMethod):
    def __init__(self,
                 model: nn.Module,
                 B: torch.tensor,
                 A: torch.tensor):

        self.model = model
        self.B = B
        self.A = A
