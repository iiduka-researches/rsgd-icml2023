import torch
import torch.nn as nn
from torch.autograd import Function


__all__ = [
    'EigLayer',
    'Matrix_Log',
    'Matrix_Sqrt'
]


class EigLayerF(Function):
    @staticmethod
    def forward(self, input: torch.Tensor):
        dim: int = input.dim()
        if not (dim == 2 or dim == 3):
            raise Exception(f'input.dim() is {dim}.')

        value, vector = torch.linalg.eig(input)
        L = torch.zeros_like(input)
        if dim == 2:
            L = torch.diag(value.real)
        else:
            n: int = input.shape[0]
            for idx in range(n):
                L[idx] = torch.diag(value[idx].real)
        V = vector.real

        self.save_for_backward(input, L, V)
        return L, V

    @staticmethod
    def backward(self, grad_S, grad_U):
        input, S, U = self.saved_tensors
        dim = input.shape[1]
        grad_input = torch.zeros(input.shape)
        e = torch.eye(dim)

        P_i = S @ torch.ones(dim, dim)
        
        P = (P_i - P_i.permute(0,2,1)) + e
        epo = (torch.ones(P.shape)) * 1e-6
        P = torch.where(P != 0, P, epo)
        P = (1 / P) - e
        
        g1 = U.permute(0, 2, 1) @ grad_U
        g1 = (g1 + g1.permute(0, 2, 1)) / 2
        g1 = P.permute(0, 2, 1) * g1
        g1 = 2 * U @ g1 @ U.permute(0, 2, 1)
        g2 = U @ (grad_S * e) @ U.permute(0, 2, 1)
        grad_input = g1 + g2

        return grad_input


class EigLayer(nn.Module):
    def __init__(self) -> None:
        super(EigLayer, self).__init__()
    
    def forward(self, input1):
        return EigLayerF().apply(input1)


class Matrix_Log(nn.Module):
    def __init__(self):
        super(Matrix_Log, self).__init__()
        self.beta = 1e-6

    def forward(self, input):
        dim = input.shape[1]

        espison = torch.eye(dim) * self.beta
        espison = torch.unsqueeze(espison, 0)
        input2 = torch.where(input - espison < 0, espison, input)
        one = torch.ones(input2.shape)
        e = torch.eye(dim)
        output = torch.log(input2 + one - e)

        return output


class Matrix_Sqrt(nn.Module):
    def __init__(self, sign: int) -> None:
        super(Matrix_Sqrt, self).__init__()
        self.sign = sign
        self.epsilon = 1e-6

    def forward(self, input):
        dim = input.shape[1]

        one = torch.ones(input.shape)
        e = torch.eye(dim)

        output = input + one - e
        output = torch.where(output > 0, output, self.epsilon*one)
        output = torch.sqrt(output)
        if self.sign == -1:
            output = 1 / output
        output = output - one + e

        return output
