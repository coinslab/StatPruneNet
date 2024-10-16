from torch.nn.utils.prune import BasePruningMethod
import torch

class StatisticalSensitivityPrune(BasePruningMethod):
    PRUNING_TYPE = 'structured'

    def __init__(self, units_to_prune):
        self.units_to_prune = units_to_prune

    def compute_mask(self, tensor, default_mask):
        mask = default_mask.clone()

        mask = mask * self.units_to_prune

        return mask
    
def prune_layer(module, name, units_to_prune):
    StatisticalSensitivityPrune.apply(module, name, units_to_prune)

    return module