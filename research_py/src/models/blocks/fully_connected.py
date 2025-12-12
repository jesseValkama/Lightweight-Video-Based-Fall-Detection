import torch
from torch import nn


class FullyConnected(nn.Module):
    """
    NOT USED
    """
    def __init__(self, in_channels, out_channels, bias, activation_function = None) -> None:
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias)
        self.activation_function = activation_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        if self.activation_function:
            x = self.activation_function(x)
        return x