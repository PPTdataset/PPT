import torch.nn as nn

# num_groups in group norm
_GN_NUM_GROUPS = 16


def get_norm_layer(num_channels: int, norm_layer: str) -> nn.Module:
    """
    Return norm layer according to 'norm_layer'
    :param num_channels: input channels
    :param norm_layer: bn - batch norm, gn - group norm
    :return:
    """
    if norm_layer == 'bn':
        nl = nn.BatchNorm2d(num_channels)
    elif norm_layer == 'gn':
        nl = nn.GroupNorm(_GN_NUM_GROUPS, num_channels)
    else:
        raise Exception('Unknown normalization layer: {}.'.format(norm_layer))
    return nl


def get_up_sample_layer(scale_factor: float, mode: str):
    """
    Return up-sample layer according to given mode
    :param scale_factor:
    :param mode:
    :return:
    """
    if mode == 'nearest':
        return nn.Upsample(scale_factor=scale_factor, mode='nearest')
    elif mode == 'bilinear':
        return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
