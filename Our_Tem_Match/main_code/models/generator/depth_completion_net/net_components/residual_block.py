import torch.nn as nn
from .utils import *


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualUnit(nn.Module):
    """
    Residual unit for network of at least 50 layers
    """
    def __init__(self, config, stride=1, norm_layer='bn', opt=None):
        """
        Initialize
        :param config: configuration for residual block in form [input_channels (output_channels_0), output_channels_1,
        output_channels_2, ...]
        :param stride: stride of first conv layer
        """
        super(ResidualUnit, self).__init__()
        self.opt = opt
        # assert
        assert len(config) == 4
        assert stride in [1, 2]
        # convolution layers
        self.conv1 = nn.Conv2d(config[0], config[1], 1, bias=False)
        self.conv2 = nn.Conv2d(config[1], config[2], 3, padding=1, bias=False, stride=stride)
        self.conv3 = nn.Conv2d(config[2], config[3], 1, bias=False)
        # norm layers
        self.norm1 = get_norm_layer(config[1], norm_layer)
        self.norm2 = get_norm_layer(config[2], norm_layer)
        self.norm3 = get_norm_layer(config[3], norm_layer)
        # activation function
        self.relu = nn.ReLU(inplace=True)
        # down-sample
        if stride != 1 or config[0] != config[3]:
            self.down_sample = nn.Sequential(
                nn.Conv2d(config[0], config[3], 1, bias=False, stride=stride),
                get_norm_layer(config[3], norm_layer)
            )
        else:
            self.down_sample = None
        if self.opt.residual_attention:
            self.seLayer = SELayer(config[3])


    def forward(self, x):
        """
        Forward step
        :param x: input data
        :return:
        """
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        # se block
        if self.opt.residual_attention:
            out = self.seLayer(out)

        if self.down_sample is not None:
            out = out + self.down_sample(x)
        else:
            out = out + x

        return self.relu(out)


class ResidualBlock(nn.Module):
    """
    Residual block
    """
    def __init__(self, config, num_layers, stride, norm_layer, opt=None):
        """
        Initialize
        :param config: configuration of residual unit
        :param num_layers: number of layers
        :param stride: stride
        :param norm_layer: normalization layer
        """
        super(ResidualBlock, self).__init__()
        # layers
        units = []
        for i in range(num_layers):
            s = stride if i == 0 else 1
            units.append(ResidualUnit(config, stride=s, norm_layer=norm_layer, opt=opt))
            config[0] = config[-1]
        self.units = nn.Sequential(*units)

    def forward(self, x):
        """
        Forward step
        :param x: input data
        :return:
        """
        return self.units(x)
