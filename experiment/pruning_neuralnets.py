from experiment.train import Train
from experiment.cross_validation import CrossValidation
from experiment.prune import Prune
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Optimizer
from typing import Optional
import torch.linalg

class PruningNeuralNets:
    def __init__(self,
                 model:nn.Module,
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 loss_fn: nn.Module,
                 optimizer: Optimizer,
                 epochs: Optional[int] = 20,
                 threshhold: Optional[float] = 0.0007,
                 l2_lambda: float = 0.01,
                 l1_approx_lambda: float = 0.01,
                 device_name: Optional[str] = None):

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.threshhold = threshhold
        self.l2_lambda = l2_lambda
        self.l1_approx_lambda = l1_approx_lambda
        
        if device_name is None:
            print("Using CPU")
            self.device = torch.device('cpu')
        else:
            print("Using GPU")
            self.device = torch.device(device_name)
            self.model.to(self.device)


    def experiment(self) -> None:
        trainer = Train(self.model)

        self.model, B, A = trainer.train(self.train_dataset,
                                         self.loss_fn,
                                         self.optimizer,
                                         self.device,
                                         self.l2_lambda,
                                         self.l1_approx_lambda,
                                         self.threshhold)
        

        # print(torch.linalg.matrix_rank(A))
        # print(torch.linalg.pinv(A))