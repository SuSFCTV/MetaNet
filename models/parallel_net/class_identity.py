import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    @staticmethod
    def forward(x):
        return x
