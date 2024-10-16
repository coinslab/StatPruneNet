from experiment.train import Train
from experiment.cross_validation import CrossValidation
from experiment.statistical import Statistical
import torch
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    device_name: str | None = None
    epochs: int = 50
    gmin: float = 1e-3
    l2_lambda: float = 0.1
    l1_approx_lambda: float = 0.1
    epsilon: float = 1e-5
    tolerance: float = 1e-3

class PruningExperiment:
    def __init__(self,
                 model:torch.nn.Module,
                 train_dataset: torch.utils.data.Dataset,
                 test_dataset: torch.utils.data.Dataset,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 hyperparameters: Hyperparameters):

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = hyperparameters.epochs
        self.gmin = hyperparameters.gmin
        self.l2_lambda = hyperparameters.l2_lambda
        self.l1_approx_lambda = hyperparameters.l1_approx_lambda
        self.epsilon = hyperparameters.epsilon
        self.tolerance = hyperparameters.tolerance
        self.len_dataset = len(train_dataset)
        
        if hyperparameters.device_name is None:
            print("Using CPU")
            self.device = torch.device('cpu')
        else:
            print("Using GPU")
            self.device = torch.device(hyperparameters.device_name)
            self.model.to(self.device)

    def experiment(self) -> None:
        trainer = Train(self.model)

        self.model, B, A = trainer.train(train_dataset=self.train_dataset,
                                         loss_fn=self.loss_fn,
                                         optimizer=self.optimizer,
                                         epochs=self.epochs,
                                         device=self.device,
                                         len_dataset=self.len_dataset,
                                         gmin=self.gmin,
                                         l2_lambda=self.l2_lambda,
                                         l1_approx_lambda=self.l1_approx_lambda)
        
        
        print("Statistical Pruning on trained model...\n")
        statistical_pruning = Statistical(self.model, self.epsilon, self.tolerance)

        statistical_pruning.prune(B, A, self.len_dataset)
        # statistical_pruning.prune(B, B, self.len_dataset)
        # statistical_pruning.prune(A, A, self.len_dataset)

        """
        Args:
            l2_lambda (float): 
            len_dataset (int): 

        Attributes:
            eps (float):

        Returns:
            torch.Tensor: .

        Raises:
            AssetionError:
        """