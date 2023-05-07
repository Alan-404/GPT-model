import torch
import torch.nn as nn
import torch.nn.functional as F


activations = {
    'relu': F.relu,
    "gelu": F.gelu,
    "tanh": torch.tanh,
    'lrelu' or "leaky_relu" or "leakyrelu": F.leaky_relu,
    "sigmoid": F.sigmoid,
    "softmax": F.softmax
}

def get_device(device: str) -> torch.device:
    if device != "cpu" and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")

