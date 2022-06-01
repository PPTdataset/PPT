import torch.nn as nn
from .utils import *

class BasicConv2d(nn.Module):
    """
    A basic 2d convolution layer, processing conv, batchnorm and relu.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1,
                 norm_layer=None):
        """
        Initialize
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: kernel size
        :param stride: stride size
        :param padding: padding size
        :param dilation: dilation for conv
        :param norm_layer: normalization layer, bn - batch norm; gn - group norm
        """
        super(BasicConv2d, self).__init__()
        # check norm layer
        if norm_layer is None:
            raise Exception('The normalization layer has not been given.')
        # layers
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False)
        self.norm = get_norm_layer(out_channels, norm_layer)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        """
        Forward step
        :param x: input data
        :return:
        """
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)

        return out
