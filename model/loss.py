import torch
from torch import Tensor
import torch.nn as nn
class Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs: Tensor, labels: Tensor) -> Tensor:
        batch_size = labels.size(0)

        loss = 0.0
        for batch in range(batch_size):
            loss += self.criterion(outputs[batch], labels[batch])
        return loss/batch_size