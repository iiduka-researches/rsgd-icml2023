import os
from tqdm import tqdm

import torch
import torch.nn as nn
import pandas as pd

from kylberg import Kylberg

from modules import (
    EigLayer,
    Matrix_Sqrt,
    Matrix_Log
)

from leaningrate import (
    ConstantLR,
    DiminishingLR,
    MultiStepLR
)


class Retraction(nn.Module):
    def __init__(self) -> None:
        super(Retraction, self).__init__()
    
    def forward(self, input: torch.Tensor, grad: torch.Tensor) ->  torch.Tensor:
        m_inv = torch.linalg.inv(input)
        output = input + grad + 0.5 * grad @ m_inv @ grad
        return output


def f(input, M) -> float:
    eiglayer1 = EigLayer()
    eiglayer2 = EigLayer()
    msqrt = Matrix_Sqrt(-1)
    mlog = Matrix_Log()
    
    A_L, A_V = eiglayer1(input)
    A_L1 = msqrt(A_L)
    A_1 = A_V @ A_L1 @ A_V.permute(0, 2, 1)
    A_e = A_1 @ M @ A_1
    A_eL, A_eV = eiglayer2(A_e)
    ALeL = mlog(A_eL)
    A_log = A_eV @ ALeL @ A_eV.permute(0, 2, 1)
    output = torch.linalg.norm(A_log, axis=1)
    output = torch.pow(output, 2)
    loss = torch.sum(output)

    return loss


def evaluate_loss(M: torch.Tensor, test_loader) -> float:
    running_loss: float = 0.
    total: int = 0
    for input, _ in test_loader:
        loss = f(input, M)
        running_loss += loss.item()
        total += input.shape[0]
    return running_loss / total


if __name__ == '__main__':
    lr_type: str = 'constant'
    alpha: float = 5e-4
    max_epoch: int = 200
    epsilon = 0.25
    result_dir: str = 'result'
    
    path: str = os.path.join(result_dir, lr_type)
    if not os.path.isdir(path):
        os.makedirs(path)

    train_data = Kylberg('data', train=True)
    M0: torch.Tensor = torch.eye(5)
    retraction: nn.Module = Retraction()

    for b in tqdm(range(2 ** 5, 2 ** 9 + 1)):
        if lr_type == 'constant':
            lr = ConstantLR(alpha=alpha)
        elif lr_type == 'diminishing1':
            lr  = DiminishingLR(alpha=alpha, p=0.5)
        elif lr_type == 'diminishing2':
            lr = MultiStepLR(alpha=alpha, gamma=0.9, milestones=[k for k in range(100)])
        else:
            raise ValueError(f'Invaild learning rate type: {lr_type}')

        loss_list: list[float] = []

        M: torch.Tensor = M0.detach()
        M.requires_grad = True

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=b)

        is_converge = False
        for epoch in range(max_epoch):
            for input, _ in train_loader:
                # ===== Evaluate loss =====
                if is_converge:
                    break
                el = evaluate_loss(M, train_loader)
                if el < epsilon:
                    is_converge = True
                loss_list.append(el)
                print(el)

                loss: torch.Tensor = f(input, M)

                loss.backward()    # Compute a Euclidean gradient
                nabla = 0.5 * M @ (M.grad + M.grad.T) @ M    # Compute a Riemannian gradient

                M = retraction(M, -lr() * nabla)
                M = M.detach()
                M.requires_grad = True
            
            lr.step()
    
        df = pd.DataFrame(loss_list)
        df.to_csv(os.path.join(path, f'rsgd{b}.csv'), header=None, index=None)
