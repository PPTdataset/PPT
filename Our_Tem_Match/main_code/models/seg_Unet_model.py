import torch
from .base_model import BaseModel
from . import networks
import torch.nn as nn
# from apex import amp
import os
from util.util import tensor2im, tensor2im_check
from tools.utils import Combine_img, Save

EPS = 1e-5


class DiceLoss(nn.Module):
    # @amp.float_function
    def __init__(self, loss_weight=1.0, eps=1e-5):
        super(DiceLoss, self).__init__()
        self.loss_weight = loss_weight

        # TODO: eps
        self.eps = eps
        print("diceloss_eps =", self.eps)

    # @amp.float_function
    def forward(self, input, target, mask, reduce=True):
        batch_size = input.size(0)
        # input = torch.sigmoid(input)

        input = input.contiguous().view(batch_size, -1)
        target = target.contiguous().view(batch_size, -1).float()
        mask = mask.contiguous().view(batch_size, -1).float()

        # TODO: 让黑边也学0
        input = input * mask
        target = target * mask

        target_max, _ = target.max(dim=1, keepdim=True)
        eps = (1 - target_max) * self.eps

        a = torch.sum(input * target, dim=1)
        b = torch.sum(input * input, dim=1)
        c = torch.sum(target * target, dim=1)
        # d = (2 * a + self.eps) / (b + c + self.eps)
        d = (2 * a + eps) / (b + c + eps)

        loss = 1 - d
        loss = self.loss_weight * loss
        if reduce:
            loss = torch.mean(loss)

        return loss


def L4_loss(input, target, reduction='mean'):
    ret = (input - target)**4
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


class SegUnetModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.set_defaults(pool_size=0)
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.loss_names = ['DiceLoss']
        self.model_names = ['G']

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      opt.is_sigmoid, self.isTrain, opt)

        if self.isTrain:
            # self.criterion = DiceLoss(eps=opt.dice_loss_eps)
            self.criterion = torch.nn.BCELoss()

            self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
            self.netG = self.netG.to(self.gpu_ids[0])
            # self.netG, self.optimizer_G = amp.initialize(self.netG, self.optimizer_G, opt_level=opt.opt_level)
            # self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)  # multi-GPUs
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.OK_img = input['OK_img']
        self.DF_img = input['DF_img']
        self.Label = input['Label']
        self.Mask = input['Mask']
        self.Input = input['Input']
        self.Match_img = input['Match_img']

    def forward(self):
        self.Pred_img = self.netG(self.Input)

    def backward_G(self):
        # self.loss_DiceLoss = self.criterion(self.Pred_img, self.Label, self.Mask) * self.opt.lambda_loss_L1
        self.loss_DiceLoss = self.criterion(self.Pred_img, self.Label) * self.opt.lambda_loss_L1

        # with amp.scale_loss(self.loss_DiceLoss, self.optimizer_G) as scaled_loss:
        #     scaled_loss.backward()
        self.loss_DiceLoss.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def get_current_visuals(self):
        pred = self.Pred_img.clone() * self.Mask
        pred[pred > 0.5] = 1.
        pred[pred <= 0.5] = 0.
        return self.Input, pred,self.Pred_img, self.Label

    def get_miou(self):
        pred = self.Pred_img.new_zeros(self.Pred_img.size())
        pred[self.Pred_img * self.Mask > 0.5] = 1.
        label = self.Label

        if 1:
            # print(pred.size())
            pred = pred.reshape(pred.size(0), -1)
            label = label.reshape(pred.size(0), -1)
            inter = ((pred * label) > 0).sum(1)
            union = ((pred + label) > 0).sum(1)
            miou = ((inter.float() + EPS) / (union.float() + EPS)).mean()
        else:
            inter = ((pred * label) > 0).sum()
            union = ((pred + label) > 0).sum()
            miou = 1.0 * inter / union
        return miou

    def save_check(self, name, save_check_dir):
        # 读图
        image_list = self.get_current_visuals()
        pred = self.Pred_img.new_zeros(self.Pred_img.size())
        pred[self.Pred_img * self.Mask > 0.5] = 1.
        #pred=self.Pred_img
        label = self.Label
        pred = pred.reshape(pred.size(0), -1)
        label = label.reshape(pred.size(0), -1)
        inter = ((pred * label) > 0).sum(1)
        union = ((pred + label) > 0).sum(1)
        batch_miou = (inter.float() + EPS) / (union.float() + EPS)
        data, ind = batch_miou.topk(pred.size(0), dim=0)

        image_list2 = [tensor2im_check(img[ind], is_sigmoid=self.opt.is_sigmoid) for img in image_list]
        Save(Combine_img(self.opt.input_nc, image_list2,data), name, path=save_check_dir)

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                torch.save(net.state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                if self.opt.load_model_from_models:
                    load_path = self.load_path
                else:
                    load_filename = '%s_net_%s.pth' % (epoch, name)
                    load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)

                if self.opt.print_log:
                    print('loading the model from %s' % load_path)

                state_dict = torch.load(load_path, map_location=str(self.device))
                net.load_state_dict(state_dict)