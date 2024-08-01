from torch.func import functional_call, grad, vmap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from typing import Optional, Tuple
import gc
import logging

class Train:
    def __init__(self, model: nn.Module):
        self.model = model

    def _l2_regularization(self, l2_lambda: float, len_dataset: int) -> torch.Tensor:
        l2_lambda = l2_lambda / (2 * len_dataset)
        l2_norm = sum(torch.norm(p, p=2)**2 for p in self.model.parameters())
        loss_l2 = l2_lambda * l2_norm

        return loss_l2

    def _log_cosh_regularization(self, l1_approx_lambda: float, len_dataset: int) -> torch.Tensor:
        l1_approx_lambda = l1_approx_lambda / len_dataset
        l1_approx_sum = sum(torch.log(torch.cosh(2.3099 * p)).sum() for p in self.model.parameters())
        loss_l1_approx = l1_approx_lambda * l1_approx_sum
 
        return loss_l1_approx
    
    '''Highly vectorized, high memory usage'''
    def _compute_A(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        def compute_loss(params, x, y):
            y_hat = functional_call(self.model, params, x)
            loss = F.cross_entropy(y_hat, y)
            return loss
        
        params = {k: v.detach() for k, v in self.model.named_parameters()}

        H = torch.func.hessian(compute_loss)(params, x, y)

        param_sizes = [param.numel() for param in self.model.parameters()]

        tensor_list = []
        k = 0

        for key in H:
            param_list = []
            key_param = list(H[key].values())

            for i in range(len(key_param)):
                param_list.append(key_param[i].view(param_sizes[k], -1))

            tensor_list.append(torch.cat(param_list, dim=1))
            k += 1

        A = torch.cat(tensor_list, dim=0)

        return A
        
    '''Basic brute force, takes a while to compute but way less memory usage'''
    def _compute_A1(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)

        grad_vector = torch.cat([g.view(-1) for g in grads])

        hessian = []

        for grad_i in grad_vector:
            grad2 = torch.autograd.grad(grad_i, self.model.parameters(), retain_graph=True)

            hessian_row = torch.cat([g.contiguous().view(-1) for g in grad2])
            hessian.append(hessian_row)

        hessian = torch.stack(hessian)
        
        return hessian

    def _compute_G(self, x: torch.Tensor, y: torch.Tensor, len_dataset: int) -> Tuple[torch.Tensor, float]:
        def compute_loss(params, x, y):
            x = x.clone().unsqueeze(0)
            y = y.clone().unsqueeze(0)
            y_hat = functional_call(self.model, params, x)
            loss = F.cross_entropy(y_hat, y)

            return loss

        params = {k: v.detach() for k, v in self.model.named_parameters()}
        grad_loss = grad(compute_loss)
        all_grads = vmap(grad_loss, in_dims=(None, 0, 0))
        grads = all_grads(params, x, y)

        grads = list(grads.values())
        G = torch.cat([g.view(len_dataset, -1) for g in grads], dim=1)
        gradmax = G.mean(dim=0).abs().max().item()


        return G, gradmax

    def _empty_cache(self, device: torch.device) -> None:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

    def train(self,
              train_dataset: Dataset,
              loss_fn: nn.Module,
              optimizer: Optimizer,
              device: torch.device,
              l2_lambda: Optional[float] = 0.1,
              l1_approx_lambda: Optional[float] = 0.1,
              threshhold: Optional[float] = 0.0007) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
        
        len_dataset = len(train_dataset)
        gradmax = float('inf')
        
        print("Training")
        self.model.train()

        trainloader = DataLoader(train_dataset, batch_size=len_dataset, shuffle=True, num_workers=0, pin_memory=True)
        x, y = next(iter(trainloader))
        x, y = x.to(device), y.to(device)

        epoch = 0
        while (gradmax >= threshhold):
            def closure():
                optimizer.zero_grad()
                y_hat = self.model(x)       
                loss = loss_fn(y_hat, y) + self._l2_regularization(l2_lambda, len_dataset) + self._log_cosh_regularization(l1_approx_lambda, len_dataset)
                loss.backward()

                return loss

            loss = optimizer.step(closure)
            G, gradmax = self._compute_G(x, y, len_dataset)

            print(f"\n\tEpoch = {epoch + 1}\tTraining loss = {loss:.4f}\tGradmax = {gradmax:.4f}")
            epoch += 1

        print("GRADMAX value achieved!")
        print("Training complete!")
        G = G.cpu()
        B = torch.matmul(G.T, G)

        del G

        A = self._compute_A1(x, y) 

        del trainloader, x, y

        gc.collect()
        self._empty_cache(device)

        return self.model, B.to(device), A.to(device)
