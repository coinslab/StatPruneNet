from experiment.train import Train
from experiment.cross_validation import CrossValidation
from experiment.wald import Wald
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Optimizer
from typing import Optional

class PruningNeuralNets:
    def __init__(self,
                 model:nn.Module,
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 loss_fn: nn.Module,
                 optimizer: Optimizer,
                 epochs: Optional[int] = 20,
                 threshold: Optional[float] = 1e-4,
                 ep: Optional[float] = 1e-5,
                 l2_lambda: float = 0.1,
                 l1_approx_lambda: float = 0.1,
                 device_name: Optional[str] = None):

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.ep = ep
        self.threshold = threshold
        self.l2_lambda = l2_lambda
        self.l1_approx_lambda = l1_approx_lambda
        self.len_dataset = len(train_dataset)
        
        if device_name is None:
            print("Using CPU")
            self.device = torch.device('cpu')
        else:
            print("Using GPU")
            self.device = torch.device(device_name)
            self.model.to(self.device)

    def experiment(self) -> None:
        trainer = Train(self.model)

        self.model, B, A = trainer.train(train_dataset=self.train_dataset,
                                         loss_fn=self.loss_fn,
                                         optimizer=self.optimizer,
                                         device=self.device,
                                         len_dataset=self.len_dataset,
                                         threshold=self.threshold,
                                         l2_lambda=self.l2_lambda,
                                         l1_approx_lambda=self.l1_approx_lambda)
        
        
        print("Going on to the Wald test on trained model...\n")
        wald = Wald(self.model)

        wald.prune(B, A, self.len_dataset)

"""
    class ModelArgs:
        dim: int = 4096
        n_layers: int = 32
        n_heads: int = 32
        n_kv_heads: Optional[int] = None
        vocab_size: int = -1  # defined later by tokenizer
        multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
        ffn_dim_multiplier: Optional[float] = None
        norm_eps: float = 1e-5

        max_batch_size: int = 32
        max_seq_len: int = 2048

    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
"""