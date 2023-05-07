import torch
import numpy as np


def generate_padding_mask(tensor: torch.Tensor)-> torch.Tensor:
    return torch.Tensor(tensor == 0).type(torch.int64)[:, np.newaxis, np.newaxis, :]

def __generate_look_ahead_mask(length: int) -> torch.Tensor:
    return torch.triu(torch.ones((length, length)), diagonal=1)

def generate_look_ahead_mask(tensor: torch.Tensor) -> torch.Tensor:
    padding_mask = generate_padding_mask(tensor)

    look_ahead_mask = __generate_look_ahead_mask(tensor.size(1)).to(tensor.device)


    look_ahead_mask = torch.maximum(look_ahead_mask, padding_mask)

    return look_ahead_mask