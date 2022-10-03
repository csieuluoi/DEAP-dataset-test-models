import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GraphCN(nn.Module):
    """
    A simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, X, A):
        """
        A: adjecency matrix
        X: graph signal
        """
        x = self.linear(X)
        out = torch.einsum("c c, b c x -> b c x", A, x)
        # out = torch.bmm(A, x)
        return out
