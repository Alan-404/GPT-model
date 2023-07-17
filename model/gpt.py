import torch
from torch import Tensor
import torch.nn as nn
from .components.decoder import Decoder
from .components.embedding import TextAndPositonEmbeded
from .utils.mask import generate_look_ahead_mask
from typing import Union, Callable

class GPT(nn.Module):
    def __init__(self, token_size: int, n: int, d_model: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[Tensor], Tensor]]) -> None:
        super().__init__()
        self.embedding = TextAndPositonEmbeded(token_size ,d_model)
        self.decoder = Decoder(n, d_model, heads, d_ff, dropout_rate, eps, activation)
        self.classifier = nn.Linear(in_features=d_model, out_features=token_size)
    
    def forward(self, x: Tensor) -> Tensor:
        if self.training == False:
            self.train()
        mask = generate_look_ahead_mask(x)
        x = self.embedding(x)
        x = self.decoder(x, mask)
        x = self.classifier(x)
        return x
    
    def inference(self, x: Tensor) -> Tensor:
        if self.training == True:
            self.eval()
        x = self.embedding(x)
        x = self.decoder(x, None)
        x = self.classifier(x)
        return x
    
    def predict(self, x: Tensor, num_tokens: int, end_token: int) -> Tensor:
        if self.training == True:
            self.eval()
        for _ in range(num_tokens):
            output = self.inference(x)

            preds = output[:, -1, :]
            _, predict_token = torch.max(preds, dim=-1)
            
            if predict_token == end_token:
                break
            x = torch.concat([x, predict_token.unsqueeze(0)], dim=-1)
        return x