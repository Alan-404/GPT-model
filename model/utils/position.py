import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def __encode_ctx(self, n_ctx: int) -> Tensor:
        pos = torch.arange(n_ctx)
        pos = pos.unsqueeze(-1)
        return pos.type(torch.float32)
    
    def __encode_embedding(self, embedding_dim: int) -> Tensor:
        angles = torch.arange(embedding_dim)
        angles[1::2] = angles[0::2]
        angles = 1/(torch.pow(10000, angles/embedding_dim))
        angles = angles.unsqueeze(0)
        return angles
    
    def forward(self, x: Tensor) -> Tensor:
        pos = self.__encode_ctx(x.size(1))
        angles = self.__encode_embedding(x.size(2))
        
        pos_angles = torch.matmul(pos, angles)
        pos_angles[0::2] = torch.sin(pos_angles[0::2])
        pos_angles[1::2] = torch.cos(pos_angles[1::2])

        pos_angles = pos_angles.unsqueeze(0)
        x += pos_angles.to(x.device)
        return x

