
import torch

from torch import nn

class DLinear(nn.Module):
    def __init__(self, input_size, seasonality_size, trend_size):
        super().__init__()
        self.seasonality_layer = nn.Linear(input_size, seasonality_size)
        self.trend_layer = nn.Linear(input_size, trend_size)

    def forward(self, x):
        pass
        