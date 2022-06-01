import torch.nn as nn
import torch
from .net_components.resnet_backbone import resnet50_backbone, resnet34_backbone
from .net_components.residual_block import ResidualBlock, SELayer
from .net_components.basic_conv import BasicConv2d
# from cspn import AffinityPropagate
# from common import configurations


# normalization layer
_NORM_LAYER = 'bn'
# align corner
_ALIGN_CORNER = True
# depth invalid value
# _DEPTH_INVALID_VALUE = configurations.depth_invalid_value


# print
# print('Align corner: {}.'.format(_ALIGN_CORNER))


def get_norm_layer(in_channels):
    return nn.GroupNorm(16, in_channels)


class _C1(nn.Module):
    def __init__(self, layer4, opt=None):
        super(_C1, self).__init__()
        # layers
        self.opt = opt
        self.block1 = layer4
        self.block2 = ResidualBlock([2048, 256, 256, 1024], opt.num_residual_layer[2], 1, _NORM_LAYER, opt)
        self.skip_block = ResidualBlock([1024, 256, 256, 1024], opt.num_residual_layer[2], 1, _NORM_LAYER, opt)
        self.up_sample = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=_ALIGN_CORNER)
        if self.opt.C123_attention:
            self.seLayer = SELayer(1024)

    def forward(self, x):
        # print("C1", x.shape)
        out = self.block1(x)
        out = self.up_sample(out)
        out = self.block2(out)
        if self.opt.C123_attention:
            out = self.seLayer(out)
        out = out + self.skip_block(x)
        return self.up_sample(out)


class _C2(nn.Module):
    def __init__(self, layer3, layer4, opt=None):
        super(_C2, self).__init__()
        # layers
        self.opt = opt
        self.block1 = layer3
        self.block2 = _C1(layer4, opt)
        self.block3 = ResidualBlock([1024, 128, 128, 512], opt.num_residual_layer[1], 1, _NORM_LAYER, opt)
        self.skip_block = ResidualBlock([512, 128, 128, 512], opt.num_residual_layer[1], 1, _NORM_LAYER, opt)
        self.up_sample = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=_ALIGN_CORNER)
        if self.opt.C123_attention:
            self.seLayer = SELayer(512)

    def forward(self, x):
        # print("C2", x.shape)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        if self.opt.C123_attention:
            out = self.seLayer(out)            
        out = out + self.skip_block(x)
        return self.up_sample(out)


class _C3(nn.Module):
    def __init__(self, layer2, layer3, layer4, opt=None):
        super(_C3, self).__init__()
        # layers
        self.opt = opt
        self.block1 = layer2
        self.block2 = _C2(layer3, layer4, opt)
        self.block3 = ResidualBlock([512, 64, 64, 256], opt.num_residual_layer[0], 1, _NORM_LAYER, opt)
        self.skip_block = ResidualBlock([256, 64, 64, 256], opt.num_residual_layer[0], 1, _NORM_LAYER, opt)
        self.up_sample = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=_ALIGN_CORNER)
        if self.opt.C123_attention:
            self.seLayer = SELayer(256)

    def forward(self, x):
        # print("C3", x.shape)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        if self.opt.C123_attention:
            out = self.seLayer(out)
        out = out + self.skip_block(x)
        return self.up_sample(out)


class C1(nn.Module):
    def __init__(self, layer4, opt=None):
        super(_C1, self).__init__()
        # layers
        self.opt = opt
        self.block1 = layer4
        self.block2 = ResidualBlock([2048, 256, 256, 1024], opt.num_residual_layer[2], 1, _NORM_LAYER, opt)
        self.skip_block = ResidualBlock([1024, 256, 256, 1024], opt.num_residual_layer[2], 1, _NORM_LAYER, opt)
        self.up_sample = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=_ALIGN_CORNER)
        if self.opt.C123_attention:
            self.seLayer = SELayer(1024)

    def forward(self, x):
        # print("C1", x.shape)
        out = self.block1(x)
        out = self.up_sample(out)
        out = self.block2(out)
        if self.opt.C123_attention:
            out = self.seLayer(out)
        out = out + self.skip_block(x)
        return self.up_sample(out)


class _C2(nn.Module):
    def __init__(self, layer3, layer4, opt=None):
        super(_C2, self).__init__()
        # layers
        self.opt = opt
        self.block1 = layer3
        self.block2 = _C1(layer4, opt)
        self.block3 = ResidualBlock([1024, 128, 128, 512], opt.num_residual_layer[1], 1, _NORM_LAYER, opt)
        self.skip_block = ResidualBlock([512, 128, 128, 512], opt.num_residual_layer[1], 1, _NORM_LAYER, opt)
        self.up_sample = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=_ALIGN_CORNER)
        if self.opt.C123_attention:
            self.seLayer = SELayer(512)

    def forward(self, x):
        # print("C2", x.shape)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        if self.opt.C123_attention:
            out = self.seLayer(out)            
        out = out + self.skip_block(x)
        return self.up_sample(out)


class C3(nn.Module):
    def __init__(self, layer2, layer3, layer4, opt=None):
        super(C3, self).__init__()
        # layers
        self.opt = opt
        self.block1 = layer2
        self.block2 = C2(layer3, layer4, opt)
        self.block3 = ResidualBlock([512, 64, 64, 256], opt.num_residual_layer[0], 1, _NORM_LAYER, opt)
        self.skip_block = ResidualBlock([256, 64, 64, 256], opt.num_residual_layer[0], 1, _NORM_LAYER, opt)
        self.up_sample = nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=_ALIGN_CORNER)
        if self.opt.C123_attention:
            self.seLayer = SELayer(256)

    def forward(self, x):
        # print("C3", x.shape)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        if self.opt.C123_attention:
            out = self.seLayer(out)
        out = out + self.skip_block(x)
        return self.up_sample(out)


class DepthRes50(nn.Module):
    """
    Hourglass network with resnet-50 backbone
    """
    def __init__(self, in_channels, out_channels_depth, out_channels_affinity=3, pre_trained=False):
        """
        Initialize
        :param in_channels: input channels
        :param out_channels_depth: channels of depth output
        :param out_channels_affinity: channels of affinity output
        :param pre_trained: whether to load pre-trained model
        """
        super(DepthRes50, self).__init__()

        resnet = resnet50_backbone(pre_trained)

        # remove last pool and fc layers
        resnet.avgpool = None
        resnet.fc = None
        # handle in_channels
        # if in_channels != 3:
        #     resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1 = nn.Conv2d(in_channels*2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # block1
        self.block1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )
        # block2
        self.block2 = _C3(resnet.layer2, resnet.layer3, resnet.layer4)
        # block for depth prediction
        self.depth_block = nn.Sequential(
            BasicConv2d(256, 128, 3, padding=1, norm_layer=_NORM_LAYER),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=_ALIGN_CORNER),
            nn.Conv2d(128, out_channels_depth, 3, padding=1)
        )
        self.sigmoid_out = nn.Sigmoid()
        # block for affinity prediction
        # self.affinity_block = nn.Sequential(
        #     BasicConv2d(256, 128, 3, padding=1, norm_layer=_NORM_LAYER),
        #     nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=_ALIGN_CORNER),
        #     nn.Conv2d(128, out_channels_affinity, 3, padding=1)
        # )
        # cspn module
        # self.cspn = AffinityPropagate(24, 3)

    # def forward(self, x: torch.Tensor, raw: torch.Tensor):
    def forward(self, x):
        """
        Forward step
        :param x: input data
        :param raw: raw depth image, in which missing depth value is presented by zero
        :return:
        """
        # get mask
        # valid_mask = (raw != _DEPTH_INVALID_VALUE).float()
        # missing_mask = (raw == _DEPTH_INVALID_VALUE).float()
        # get output
        out = self.block1(x)
        out = self.block2(out)
        # get depth output
        depth_out = self.depth_block(out)
        depth_out = self.sigmoid_out(depth_out)
        # get affinity output
        # affinity_out = self.affinity_block(out)
        # cspn out
        # cspn_out = self.cspn(affinity_out, depth_out * missing_mask + raw * valid_mask, valid_mask)
        # return
        # return depth_out, affinity_out, cspn_out
        return depth_out


############### SegV10 ###############
class SegPlus(nn.Module):
    """
    Hourglass network with resnet-50 backbone
    """
    def __init__(self, in_channels, out_channels_depth, out_channels_affinity=3, pre_trained=False, opt=None):
        """
        Initialize
        :param in_channels: input channels
        :param out_channels_depth: channels of depth output
        :param out_channels_affinity: channels of affinity output
        :param pre_trained: whether to load pre-trained model
        """
        super(SegPlus, self).__init__()

        resnet = resnet50_backbone(opt, pre_trained)
        self.opt = opt
        # remove last pool and fc layers
        resnet.avgpool = None
        resnet.fc = None
        # handle in_channels
        # if in_channels != 3:
        #     resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # block1
        self.block1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            self.conv2,
            self.bn2,
            resnet.relu,
            # resnet.maxpool,
            resnet.layer1
        )
        # block2
        self.block2 = _C3(resnet.layer2, resnet.layer3, resnet.layer4, self.opt)
        # block for depth prediction
        self.depth_block = nn.Sequential(
            BasicConv2d(256, 128, 3, padding=1, norm_layer=_NORM_LAYER),
            # nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=_ALIGN_CORNER),
            nn.Conv2d(128, out_channels_depth, 3, padding=1)
        )
        self.sigmoid_out = nn.Sigmoid()


    def forward(self, x):
        """
        Forward step
        :param x: input data
        :param raw: raw depth image, in which missing depth value is presented by zero
        :return:
        """
        # print(x.shape)
        out = self.block1(x)
        # print(out.shape)
        out = self.block2(out)
        # print(out.shape)
        # print("------------------------------")
        depth_out = self.depth_block(out)
        depth_out = self.sigmoid_out(depth_out)

        return depth_out

class SegUnet(nn.Module):
    """
    Hourglass network with resnet-50 backbone
    """
    def __init__(self, in_channels, out_channels_depth, out_channels_affinity=3, pre_trained=False, opt=None):
        """
        Initialize
        :param in_channels: input channels
        :param out_channels_depth: channels of depth output
        :param out_channels_affinity: channels of affinity output
        :param pre_trained: whether to load pre-trained model
        """
        super(SegUnet, self).__init__()

        resnet = resnet50_backbone(opt, pre_trained)
        self.opt = opt
        # remove last pool and fc layers
        resnet.avgpool = None
        resnet.fc = None
        # handle in_channels
        # if in_channels != 3:
        #     resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # block1
        self.block1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            self.conv2,
            self.bn2,
            resnet.relu,
            # resnet.maxpool,
            resnet.layer1
        )
        # block2 #此处改为Unet实现
        self.block2 = C3(resnet.layer2, resnet.layer3, resnet.layer4, self.opt)
        
        # block for depth prediction
        self.depth_block = nn.Sequential(
            BasicConv2d(256, 128, 3, padding=1, norm_layer=_NORM_LAYER),
            # nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=_ALIGN_CORNER),
            nn.Conv2d(128, out_channels_depth, 3, padding=1)
        )
        self.sigmoid_out = nn.Sigmoid()


    def forward(self, x):
        """
        Forward step
        :param x: input data
        :return:
        """
        # print(x.shape)
        out = self.block1(x)
        # print(out.shape)
        out = self.block2(out)
        # print(out.shape)
        # print("------------------------------")
        depth_out = self.depth_block(out)
        depth_out = self.sigmoid_out(depth_out)

        return depth_out