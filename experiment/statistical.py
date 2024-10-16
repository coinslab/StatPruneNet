import torch.nn as nn
import torch
from torch.linalg import matrix_rank, pinv
from torch.special import gammainc
from torch.nn.utils import parameters_to_vector
from experiment.prune import prune_layer
import logging

# 1 switch indicates the choice on lne 27,28
# 2nd switch which does the check on line 25 and line 30 (if check on line 25 or 30 fail then set p = 1)
class Statistical:
    def __init__(self, model: nn.Module, epsilon: float, tolerance: float):
        self.model = model
        self.epsilon = epsilon
        self.tolerance = tolerance    

    def create_S(self, unit_idx: int, params_per_unit: int, num_params: int) -> torch.Tensor:
        S = torch.zeros((params_per_unit, num_params))
        S[:, params_per_unit * unit_idx: params_per_unit * unit_idx + params_per_unit] = torch.eye(params_per_unit)
        
        return S

    def compute_p(self, A: torch.Tensor, B: torch.Tensor, A_pinv: torch.Tensor, S: torch.Tensor, theta: torch.Tensor, len_dataset: int) -> float:
        inf_norm = torch.max(torch.abs((S@A_pinv@A) - S).view(-1))

        #assert inf_norm <= 0.0001, 'Parameter Estimation Failure'

        C = (S@A_pinv@B@A_pinv@S.T) / len_dataset
        print(C.shape)
        print(f"Rank of matrix C: {matrix_rank(C)}")
        C_inv = pinv(C, hermitian=True, rtol=self.tolerance, atol=self.tolerance**2)
        print(f"Rank of matrix C inverse: {matrix_rank(C_inv)}")
        W = (theta.T@S.T@C_inv@S@theta).view(-1)
        r = torch.tensor([float(S.shape[0])])
        p = 1 - gammainc(r/2, W/2)

        return p
        
    def prune(self, B: dict, A: dict, len_dataset: int) -> nn.Module:
        linear_layers = [name for name, module in self.model.named_modules() if isinstance(module, nn.Linear)]

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name != linear_layers[-1]:
                #print(f'Checking layer named: {name}')

                A_layer = A[name]
                B_layer = B[name]
                print(f"Shape of A and B: {A_layer.shape[0]} x {B_layer.shape[0]}")
                print(f"Rank of matrix B: {matrix_rank(B_layer)}")
                # if (matrix_rank(B_layer) < B_layer.shape[0]):
                #     print("Locally Parameter Redundant")

                print(f"Rank of matrix A: {matrix_rank(A_layer)}")
                # if (matrix_rank(A_layer) < A_layer.shape[0]):
                #     print("Not Locally Identifiable")

                A_pinv = pinv(A_layer, hermitian=True, rtol=self.tolerance, atol=self.tolerance**2)

                num_units = module.out_features
                params_per_unit = A_layer.shape[0] // num_units

                num_params = A_layer.shape[0]
    
                theta = parameters_to_vector(module.parameters()).detach().view(num_params, -1)

                S_stacked = []

                unit_mask = []

                for u in range(num_units):
                    print(f"For unit {u + 1} in current layer.")
                    S = self.create_S(u, params_per_unit, num_params)

                    p = self.compute_p(A_layer, B_layer, A_pinv, S, theta, len_dataset)

                    if (p.item() > self.epsilon):
                        S_stacked.append(S)
                        unit_mask.append(0)
                
                if S_stacked:
                    S_stacked = torch.cat(S_stacked, dim=0)
                    p_stacked = self.compute_p(A_layer, B_layer, A_pinv, S_stacked, theta, len_dataset)

                    if (p_stacked.item() > self.epsilon):
                        unit_mask = torch.tensor(unit_mask.view(num_units, -1))

                        prune_layer(module, 'weight', unit_mask)
                        prune_layer(module, 'bias', unit_mask)

                        print(list(module.named_parameters()))
                else: 
                    print('Cannot prune!')
