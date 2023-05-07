import torch
from torch import Tensor
import torch.nn as nn
from ..utils.position import PositionalEncoding

class TextAndPositonEmbeded(nn.Module):
    def __init__(self, token_size: int, d_model: int) -> None:
        super().__init__()
        self.positional_encoding = PositionalEncoding()
        self.embedding_layer = nn.Embedding(num_embeddings=token_size, embedding_dim=d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding_layer(x)
        x = self.positional_encoding(x)
        return x