import torch.nn as nn
import torch
from torch.linalg import matrix_rank, pinv, inv
from typing import Optional
from torch.special import gammainc

# 1 switch indicates the choice on lne 27,28
# 2nd switch which does the check on line 25 and line 30 (if check on line 25 or 30 fail then set p = 1)
class Wald():
    def __init__(self, model: nn.Module):
        self.model = model

    def create_S(self, unit_idx: int, params_per_unit: int, num_params: int) -> torch.Tensor:
        S = torch.zeros((params_per_unit, num_params))
        S[:, params_per_unit * unit_idx: params_per_unit * unit_idx + params_per_unit] = torch.eye(params_per_unit)
        
        return S

    def compute_p(self, A: torch.Tensor, B: torch.Tensor, S: torch.Tensor, theta: torch.Tensor, len_dataset: int, ep: Optional[float] = 1e-5) -> float:
        tol = 1e-3

        A_pinv = pinv(A, hermitian=True, rtol=tol, atol=tol**2)

        # GET ABS MAX OF CHECK
        inf_norm = torch.max(torch.abs((S@A_pinv@A) - S))

        assert inf_norm <= 0.0001, 'Parameter Estimation Failure'
    #    C= pinvA/len_dataset, C == pinvB/len_dataset
        C = (S@A_pinv@B@A_pinv@S.T) / len_dataset
# check the rank of C
        W = (theta.T@S.T@pinv(C, hermitian=True, rtol=tol, atol=tol**2)@S@theta).view(-1)
        r = torch.tensor([float(S.shape[0])])
        p = 1 - gammainc(r/2, W/2)

        return p
        

    def prune(self, B: dict, A: dict, len_dataset: int, ep: Optional[float] = 1e-5) -> nn.Module:
        linear_layers = [name for name, module in self.model.named_modules() if isinstance(module, nn.Linear)]

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name != linear_layers[-1]:
                print(f'Checking layer named {name}')
                num_units = module.out_features

                A_layer = A[name]
                B_layer = B[name]

                if (matrix_rank(B_layer) < B_layer.shape[0]):
                    print("Locally Parameter Redundant")

                if (matrix_rank(A_layer) < A_layer.shape[0]):
                    print("Not Locally Identifiable")

                params_per_unit = A_layer.shape[0] // num_units
                num_params = A_layer.shape[0]

                params = []
                
                for param in module.parameters():
                    params.append(param.detach().view(-1))

                theta = torch.cat(params).view(num_params, -1)

                S_stacked = []

                for u in range(num_units):
                    S = self.create_S(u, params_per_unit, num_params)
                    
                    assert matrix_rank(S) ==  S.shape[0], 'Invalid Selection Function Choise'

                    p = self.compute_p(A_layer, B_layer, S, theta, len_dataset)

                    if (p.item() > ep):
                        S_stacked.append(S)

                    del S
                
                S = torch.cat(S_stacked, dim=0)

                p = self.compute_p(A_layer, B_layer, S, theta, len_dataset)
