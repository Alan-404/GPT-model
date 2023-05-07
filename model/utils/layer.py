import torch
from torch import Tensor
import torch.nn as nn
from .attention import MultiHeadAttention
from .ffn import PositionWiseFeedForwardNetworks
from .residual import ResidualConnection
from typing import Union, Callable


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[Tensor], Tensor]]) -> None:
        super().__init__()
        self.masked_multi_head_attention = MultiHeadAttention(heads, d_model)
        self.feed_forward = PositionWiseFeedForwardNetworks(d_ff, d_model, activation)

        self.residual_connection_1 = ResidualConnection(d_model, dropout_rate ,eps)
        self.residual_connection_2 = ResidualConnection(d_model, dropout_rate ,eps)

    def forward(self, x: Tensor, mask: Union[Tensor, None]) -> Tensor:
        # sublayer 1
        attention_output = self.masked_multi_head_attention(x, x, x, mask)
        attention_output = self.residual_connection_1(attention_output, x)

        # sublayer 2
        ffn_out = self.feed_forward(attention_output)
        ffn_out = self.residual_connection_2(ffn_out, attention_output)

        return ffn_out
