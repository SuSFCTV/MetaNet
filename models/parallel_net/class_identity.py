import torch
import torch.nn as nn

"""
Crutch class for layer replacement
"""


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x
