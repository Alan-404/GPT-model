import torch
from torch import Tensor
import torch.nn as nn
from typing import Union, Callable

class PositionWiseFeedForwardNetworks(nn.Module):
    def __init__(self, d_ff: int, d_model: int, activation: Union[str, Callable[[Tensor], Tensor]]) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(in_features=d_model, out_features=d_ff)
        self.activation = activation
        self.output_layer = nn.Linear(in_features=d_ff, out_features=d_model)
    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x
