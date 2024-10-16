from torch.func import functional_call, grad, vmap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import parameters_to_vector
import gc
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(filename)s - %(message)s',
    # filename='training.log'
    )

logger = logging.getLogger(__name__)

class Train:
    def __init__(self, model: nn.Module):
        self.model = model

    def _l2_regularization(self, l2_lambda: float, params: torch.Tensor) -> torch.Tensor:
        squared_sum = torch.square((parameters_to_vector(params))).sum()
        loss_l2 = l2_lambda * squared_sum

        return loss_l2

    def _log_cosh_regularization(self, l1_approx_lambda: float, params: torch.Tensor) -> torch.Tensor:
        l1_approx_sum = torch.log(torch.cosh(2.3099 * parameters_to_vector(params))).sum()
        loss_l1_approx = l1_approx_lambda * l1_approx_sum
 
        return loss_l1_approx
    
    def _compute_A(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        def compute_loss(params, x, y):
            y_hat = functional_call(self.model, params, x)
            loss = F.cross_entropy(y_hat, y)
            return loss
        
        params = {k: v.clone() for k, v in self.model.named_parameters()}

        grads = grad(compute_loss)(params, x, y)

        A = {}
        
        for name in list(params.keys())[:-2]:
            if 'weight' in name:
                layer_name = name.split('.')[0]

                layer_params = [params[f'{layer_name}.weight'], params[f'{layer_name}.bias']]

                dw = grads[f'{layer_name}.weight'].view(-1)
                db = grads[f'{layer_name}.bias'].view(-1)
                layer_grads = torch.cat([dw, db], dim=0)
                
                hessian = []

                for g in layer_grads:
                    g2 = torch.autograd.grad(g, layer_params, create_graph=True)
                    hess_row = torch.cat([h.view(-1) for h in g2])
                    hessian.append(hess_row)

                hessian = torch.stack(hessian)
  
                A[f'{layer_name}'] = hessian.detach()

        return A

    def _compute_B(self, x: torch.Tensor, y: torch.Tensor, len_dataset: int) -> dict:
        def compute_loss(params, x, y):
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            y_hat = functional_call(self.model, params, (x,))
            loss = F.cross_entropy(y_hat, y)
            return loss

        params = {k: v.detach() for k, v in self.model.named_parameters()}
        grad_loss = grad(compute_loss)
        all_grads = vmap(grad_loss, in_dims=(None, 0, 0))
        grads = all_grads(params, x, y)

        B = {}

        for name in list(params.keys())[:-2]:
            if 'weight' in name:
                layer_name = name.split('.')[0]

                dw = grads[f'{layer_name}.weight'].view(len_dataset, -1)
                db = grads[f'{layer_name}.bias'].view(len_dataset, -1)
                layer_grads = torch.cat([dw, db], dim=1).detach()

                B[f'{layer_name}'] = (layer_grads.T@layer_grads) / len_dataset
        
        return B

    def _empty_cache(self, device: torch.device) -> None:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    def train(self,
              train_dataset: Dataset,
              loss_fn: nn.Module,
              optimizer: torch.optim.Optimizer,
              epochs: int,
              device: torch.device,
              len_dataset: int,
              gmin: float = 1e-4,
              l2_lambda: float = 0.1,
              l1_approx_lambda: float = 0.1) -> tuple[nn.Module, dict, dict]:

        logger.info("Starting Training")
        self.model.train()

        trainloader = DataLoader(train_dataset, batch_size=len_dataset, shuffle=True, num_workers=0)
        x, y = next(iter(trainloader))
        x, y = x.to(device), y.to(device)

        gradmax = float('-inf')

        for epoch in range(epochs):
            def closure():
                optimizer.zero_grad()
                y_hat = self.model(x)

                params = parameters_to_vector(self.model.parameters())
                loss = loss_fn(y_hat, y) + (self._l2_regularization(l2_lambda, params) / len_dataset) + (self._log_cosh_regularization(l1_approx_lambda, params) / len_dataset)
                loss.backward()

                return loss

            loss = optimizer.step(closure)
            gradmax = max(p.grad.abs().max() for p in self.model.parameters()).item()

            logger.info(f"\n\tEpoch = {epoch + 1}\tTraining loss = {loss:.4f}\tGradmax = {gradmax:.4f}")

            if (gradmax <= gmin):
                logger.info('GRADMAX value achieved! Training complete!')
                break
            
        if (epoch >= epochs):
            logger.warning('Loop terminating without finding optimal Gradmax value!')

        B = self._compute_B(x, y, len_dataset)
        
        A = self._compute_A(x, y)
        
        del trainloader, x, y

        gc.collect()
        self._empty_cache(device)

        return self.model, B, A
