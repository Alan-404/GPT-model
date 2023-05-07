import torch
from torch import Tensor
import torch.nn as nn
from ..utils.layer import DecoderLayer

from typing import Union, Callable

class Decoder(nn.Module):
    def __init__(self, n: int, d_model: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[Tensor], Tensor]]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout_rate, eps, activation) for _ in range(n)])

    def forward(self, x: Tensor, mask: Union[Tensor, None]) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x
