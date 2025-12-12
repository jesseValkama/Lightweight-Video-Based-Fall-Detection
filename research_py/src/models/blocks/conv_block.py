import torch
from torch import nn
from typing import Tuple


class ConvBlock(nn.Module):

    def __init__(self, channels_in: int, channels_out: int, kernel_size: Tuple[int, int] = (3, 3), 
                 stride: int = 1, padding: int = 2, bias: bool = False, epsilon: int = 1e-5, activation_function = None) -> None:
        """
        Convolutional Block:
            Conv2d
            BatchNorm2d
            ActFN: None or a pytorch actfn
        Args:
            channels_in: the n of in channels
            channels_out: the n of filters
            kernel_size: the size of the kernel, default = (3,3)
            stride: the size of the stride, default = 1
            padding: the size of the padding, default = 2
            bias: whether to use bias, default = False
            epsilon: avoid 0 div error in batch norm, default = 1e-5
            activation_function: None or pytorch activation function, default = None
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding, bias=bias)
        self.batch_norm = nn.BatchNorm2d(channels_out, epsilon)
        self.activation_function = activation_function

    def forward(self, x) -> torch.Tensor:
        """
        performs the forward pass of the convolutional block
        Args:
            x: the input in NLCHW or NCHW
        returns:
            Tensor: the activations
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        if self.activation_function:
            x = self.activation_function(x)
        return x


if __name__ == "__main__":
    raise NotImplementedError("Usage: python main.py args")