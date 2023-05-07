import torch
from torch import Tensor
import torch.nn as nn
from typing import Union


class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int) -> None:
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.head_samples = int(d_model/heads)

        self.linear_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_v = nn.Linear(in_features=d_model, out_features=d_model)

        self.linear_output = nn.Linear(in_features=d_model, out_features=d_model)

    def scaled_dot_product_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Union[Tensor, None]) -> Tensor:
        dk = torch.tensor(k.size(-1))

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores/(torch.sqrt(dk))

        if mask is not None:
            attention_scores += mask*(-1e15)
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_weights, v)
    
    def split_head(self, x: Tensor) -> Tensor:
        batch_size, n_ctx, _ = x.size()
        x = x.reshape(batch_size, n_ctx, self.heads, self.head_samples)
        x = x.permute(0, 2, 1, 3)

        return x
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Union[Tensor, None]) -> Tensor:
        batch_size, n_ctx, _ = q.size()

        qw = self.linear_q(q)
        kw = self.linear_k(k)
        vw = self.linear_v(v)

        q_heads = self.split_head(qw)
        k_heads = self.split_head(kw)
        v_heads = self.split_head(vw)

        attention_output = self.scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)
        attention_output = attention_output.permute(0, 2, 1, 3)
        attention_output = attention_output.reshape(batch_size, n_ctx, self.d_model)

        return self.linear_output(attention_output)