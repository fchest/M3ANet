import torch
import torch.nn as nn
import numpy as np



class si_sidrloss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        y_true = target
        y_pred = pred
        x = y_true.squeeze()
        y = y_pred.squeeze()

        smallVal = 1e-9  # To avoid divide by zero
        a = torch.sum(y * x, dim=-1, keepdims=True) / (torch.sum(x * x, dim=-1, keepdims=True) + smallVal)

        xa = a * x
        xay = xa - y
        d = torch.sum(xa * xa, dim=-1, keepdims=True) / (torch.sum(xay * xay, dim=-1, keepdims=True) + smallVal)
        # d1=tf.zeros(d.shape)
        d1 = d == 0
        d1 = 1 - torch.tensor(d1, dtype=torch.float32)

        d = -torch.mean(10 * d1 * torch.log10(d + smallVal))
        return d
