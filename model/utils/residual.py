import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.layer_norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)
    def forward(self, x: Tensor, pre_x: Tensor) -> Tensor:
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x += pre_x
        x = self.layer_norm(x)
        return x
