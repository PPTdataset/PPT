import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from tools.utils import get_transform_location


EPS = 1e-10
class LocationModel(nn.Module):
    def __init__(self, opt, ks):
        super(LocationModel, self).__init__()

        # self.TC_img = TC_img
        self.ks = (ks, ks)
        self.conv = nn.Conv2d(opt.input_nc, 1, kernel_size=self.ks, padding=0, bias=False)
        self.iden = nn.Conv2d(opt.input_nc, 1, kernel_size=self.ks, padding=0, bias=False)
        self.iden.weight.data = torch.Tensor(np.ones([1, opt.input_nc, self.ks[0], self.ks[1]]))

    def change_weight(self, opt, path=None, img=None, if_resize=False):
        if img is None:
            TC_img = Image.open(path).convert('RGB')
        else:
            TC_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        h_ori, w_ori = TC_img.size
        w = int(w_ori / opt.resize_ratio)
        h = int(h_ori / opt.resize_ratio)
        data_transform = get_transform_location(grayscale=(opt.input_nc == 1), if_resize=if_resize, resize_shape=[w, h])
        TC_img = data_transform(TC_img).unsqueeze(0)
        self.TC_img = TC_img

        if if_resize:
            ks = (w, h)
        else:
            ks = (w_ori, h_ori)

        # assert self.TC_img.shape == torch.Tensor(np.ones([1, opt.input_nc, self.ks[0], self.ks[1]])).shape
        self.ks = ks
        self.conv = nn.Conv2d(opt.input_nc, 1, kernel_size=self.ks, padding=0, bias=False)
        self.conv.weight.data = self.TC_img
        self.iden = nn.Conv2d(opt.input_nc, 1, kernel_size=self.ks, padding=0, bias=False)
        self.iden.weight.data = torch.Tensor(np.ones([1, opt.input_nc, self.ks[0], self.ks[1]]))

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    def forward(self, x):
        out = self.conv(x)
        norm_ok = self.iden(x * x) ** 0.5
        return out / norm_ok


def get_location_model(opt, ks, if_resize=False):
    if if_resize:
        ks = int(ks / opt.resize_ratio)
    return LocationModel(opt, ks)


class MeanModel(nn.Module):
    def __init__(self, ks):
        super(MeanModel, self).__init__()

        self.ks = ks
        self.iden = nn.Conv2d(1, 1, kernel_size=self.ks, padding=0, bias=False)
        self.iden.weight = torch.nn.Parameter(torch.Tensor(np.ones([1, 1, self.ks, self.ks])))

    def forward(self, x):
        out = self.iden(x)
        return out / (self.ks * self.ks)


class GroupLocationModel(nn.Module):
    def __init__(self, opt, TC_img_list, iden_list, if_resize, ori_shape):
        super(GroupLocationModel, self).__init__()

        group_num = len(TC_img_list)
        assert group_num > 0

        h_ori, w_ori = ori_shape
        h = int(h_ori / opt.resize_ratio)
        w = int(w_ori / opt.resize_ratio)

        data_transform = get_transform_location(grayscale=(opt.input_nc == 1), if_resize=if_resize, resize_shape=[h, w])

        TC_img_weight = []
        iden_weight = []
        for TC_img, iden in zip(TC_img_list, iden_list):
            # TODO: set black hole in background to zero
            # if opt.input_nc == 3:
            #     # print(TC_img.shape)
            #     position = (TC_img.max(2) <= 10)
            #     # print(position.shape)
            #     TC_img[position] = 0
            #     iden[position] = 0

            iden = data_transform(iden).unsqueeze(0)
            iden_weight.append(iden)
            TC_img = data_transform(TC_img).unsqueeze(0)
            TC_img_weight.append(TC_img)
        
        TC_img_weight = torch.cat(TC_img_weight, 0)
        iden_weight = torch.cat(iden_weight, 0)

        if if_resize:
            ks = (h, w)
            self.conv = nn.Conv2d(opt.input_nc, group_num, kernel_size=ks, padding=0, bias=False)
            self.iden = nn.Conv2d(opt.input_nc, group_num, kernel_size=ks, padding=0, bias=False)
            self.conv.weight.data = TC_img_weight
            self.iden.weight.data = iden_weight
            # print("############# rough locating completed #############")
        else:
            ks = (h_ori, w_ori)
            self.conv = nn.Conv2d(opt.input_nc*group_num, group_num, kernel_size=ks, padding=0, bias=False, groups=group_num)
            self.iden = nn.Conv2d(opt.input_nc*group_num, group_num, kernel_size=ks, padding=0, bias=False, groups=group_num)
            self.conv.weight.data = TC_img_weight
            self.iden.weight.data = iden_weight
            self.TC_img_norm = ((TC_img_weight.reshape(group_num, -1) ** 2).sum(1) + EPS).sqrt()
            # print("############# accurate locating completed #############")


    def forward(self, x):
        out = self.conv(x)
        norm_ok = self.iden(x * x) ** 0.5
        return out / norm_ok