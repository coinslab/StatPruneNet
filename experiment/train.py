from torch.func import functional_call, grad, vmap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
import gc
import logging

class Train:
    def __init__(self, model: nn.Module):
        self.model = model

    def _l2_regularization(self, l2_lambda: float, len_dataset: int) -> torch.Tensor:
        l2_lambda_term = l2_lambda / (2 * len_dataset)
        l2_norm = sum(torch.norm(p, p=2)**2 for p in self.model.parameters())
        loss_l2 = l2_lambda_term * l2_norm

        return loss_l2

    def _log_cosh_regularization(self, l1_approx_lambda: float, len_dataset: int) -> torch.Tensor:
        l1_approx_lambda_term = l1_approx_lambda / len_dataset
        l1_approx_sum = sum(torch.log(torch.cosh(2.3099 * p)).sum() for p in self.model.parameters())
        loss_l1_approx = l1_approx_lambda_term * l1_approx_sum
 
        return loss_l1_approx
    
    def _compute_A_unvec(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        def compute_loss(params, x, y):
            y_hat = functional_call(self.model, params, x)
            loss = F.cross_entropy(y_hat, y)
            return loss
        
        params = {k: v for k, v in self.model.named_parameters()}

        grads = grad(compute_loss)(params, x, y)

        A = {}
        
        for name in list(params.keys())[:-2]:
            if 'weight' in name:
                layer_name = name.split('.')[0]

                weight = grads[f'{layer_name}.weight'].view(-1)
                bias = grads[f'{layer_name}.bias'].view(-1)
                layer = torch.cat([weight, bias], dim=0)
                
                layer_params = [params[f'{layer_name}.weight'], params[f'{layer_name}.bias']]

                hessian = []
                for g in layer:
                    grad2 = torch.autograd.grad(g, layer_params, create_graph=True, retain_graph=True)
                    hess_row = torch.cat([h.view(-1) for h in grad2])
                    hessian.append(hess_row)
                hessian = torch.stack(hessian)
                
                A[f'{layer_name}'] = hessian.detach()
                #del weight, bias, layer, layer_params, grad2, hess_row, hessian

        return A

    def _compute_B(self, x: torch.Tensor, y: torch.Tensor, device: torch.device, len_dataset: int) -> dict:
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

                weight = grads[f'{layer_name}.weight'].view(len_dataset, -1)
                bias = grads[f'{layer_name}.bias'].view(len_dataset, -1)
                layer = torch.cat([weight, bias], dim=1)

                B[f'{layer_name}'] = (layer.T@layer).detach() / len_dataset

                #del weight, bias, layer
        
        #del grads

        gc.collect()
        self._empty_cache(device)
        
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
              device: torch.device,
              len_dataset: int,
              threshold: Optional[float] = 1e-4,
              l2_lambda: Optional[float] = 0.1,
              l1_approx_lambda: Optional[float] = 0.1) -> Tuple[nn.Module, dict, dict]:

        gradmax = float('inf')
        
        print("Training")
        self.model.train()

        trainloader = DataLoader(train_dataset, batch_size=len_dataset, shuffle=True, num_workers=0, pin_memory=True)
        x, y = next(iter(trainloader))
        x, y = x.to(device), y.to(device)

        epoch = 0
        while (gradmax >= threshold):
            def closure():
                optimizer.zero_grad()
                y_hat = self.model(x)
                loss = loss_fn(y_hat, y) + self._l2_regularization(l2_lambda, len_dataset) + self._log_cosh_regularization(l1_approx_lambda, len_dataset)
                loss.backward()

                return loss

            loss = optimizer.step(closure)
            gradmax = max(p.grad.abs().max().item() for p in self.model.parameters() if p.grad is not None)

            print(f"\n\tEpoch = {epoch + 1}\tTraining loss = {loss:.4f}\tGradmax = {gradmax:.4f}")
            epoch += 1

        print("GRADMAX value achieved!")
        print("Training complete!")

        B = self._compute_B(x, y, device, len_dataset)
        
        A = self._compute_A_unvec(x, y)
        
        del trainloader, x, y

        gc.collect()
        self._empty_cache(device)

        return self.model, B, A
