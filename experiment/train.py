from torch.func import functional_call, grad, vmap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def l2(model: nn.Module) -> torch.Tensor:
    loss_l2 = sum(torch.sum(p * p) for p in model.parameters())

    return loss_l2

def log_cosh(model: nn.Module) -> torch.Tensor:
    loss_log_cosh = sum(torch.log(torch.cosh(2.3099 * p)).sum() for p in model.parameters())
    
    return loss_log_cosh

def compute_A(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    def compute_loss(params, x, y):
        y_hat = functional_call(model, params, (x,))
        loss = F.cross_entropy(y_hat, y)
        return loss
    
    params = {k: v.clone() for k, v in model.named_parameters()}

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

            A[layer_name] = hessian.detach()

    return A

def compute_B(model: nn.Module, x: torch.Tensor, y: torch.Tensor, len_dataset: int) -> dict:
    def compute_loss(params, x, y):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        y_hat = functional_call(model, params, (x,))
        loss = F.cross_entropy(y_hat, y)
        return loss

    params = {k: v.detach() for k, v in model.named_parameters()}
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

            B[layer_name] = (layer_grads.T@layer_grads) / len_dataset
    
    return B

def train(model: nn.Module,
          train_dataset: Dataset,
          criterion: nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          gmin: float,
          l2_lambda: float,
          l1_approx_lambda: float,
          train_only: bool = True) -> tuple[nn.Module, dict, dict] | nn.Module:
    
    len_dataset = len(train_dataset)
    model.train()

    trainloader = DataLoader(train_dataset, batch_size=len_dataset, shuffle=True, num_workers=4, pin_memory=True)
    x, y = next(iter(trainloader))
    x, y = x.to(device), y.to(device)

    gradmax = float('-inf')

    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            y_hat = model(x)

            loss = criterion(y_hat, y) + (l2_lambda / len_dataset) * l2(model) + (l1_approx_lambda / len_dataset) * log_cosh(model)
            loss.backward()

            return loss
        
        loss = optimizer.step(closure)
        gradmax = max(p.grad.abs().max() for p in model.parameters()).item()

        print(f'\n\tEpoch = {epoch + 1}\tTraining loss = {loss:.4f}\tGradmax={gradmax:.4f}')

        if (gradmax <= gmin):
            print('GRADMAX value achieved! Training complete!')
            break

    if (epoch >= epochs - 1):
        print('Failed to achieve GRADMAX.')

    if not train_only:
        return model

    B = compute_B(model, x, y, len_dataset)
    A = compute_A(model, x, y)

    return model, B, A    