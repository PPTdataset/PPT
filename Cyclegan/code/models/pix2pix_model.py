import torch
from .base_model import BaseModel
from . import networks
import cv2
from util.util import tensor2im


def Save(tensor, name, is_sigmoid=False):
    img = tensor2im(tensor, is_sigmoid=is_sigmoid)
    cv2.imwrite("./saves/%s.bmp" % name, img)


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.set_defaults(dataset_mode='aligned')
        if is_train:
            # parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.set_defaults(pool_size=0)
            # parser.set_defaults(pool_size=0, gan_mode='lsgan')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      opt.is_sigmoid)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        '''
        增加输入一个mask，并对self.real_A和self.real_B进行处理
        '''
        self.mask = input['mask'].to(self.device)  # -1，1或者 1，1
        '''
        mask是经过双线性差值得到resize的mask，这时候需要再对mask进行二值化
        '''
        if self.opt.is_sigmoid:
            self.mask[self.mask > 0.5] = 1.
            self.mask[self.mask <= 0.5] = 0.
            self.real_A = input['A' if AtoB else 'B'].to(self.device) * self.mask
            self.real_B = input['B' if AtoB else 'A'].to(self.device) * self.mask
        else:
            # self.mask[self.mask > 0] = 1.
            # self.mask[self.mask <=0] = 0.
            # self.mask_shift = 0 * self.mask.clone()  # -1,0
            # self.mask_shift[self.mask == 0] = -1.
            # self.real_A = input['A' if AtoB else 'B'].to(self.device) * self.mask + self.mask_shift
            # self.real_B = input['B' if AtoB else 'A'].to(self.device) * self.mask + self.mask_shift
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        '''
        用mask对self.fake_B进行处理
        '''
        if self.opt.is_sigmoid:
            self.fake_B = self.netG(self.real_A) * self.mask  # G(A)
        else:
            # self.fake_B = self.netG(self.real_A) * self.mask + self.mask_shift  # G(A)
            if self.opt.if_res:
                # TODO
                self.fake_B = 2 * self.netG(self.real_A) + self.real_A
            else:
                self.fake_B = self.netG(self.real_A)  # G(A)

        # save images
        # if self.opt.save_img:
        if 0:
            self.mask_ = self.mask.clone()
            if not self.opt.is_sigmoid:
                self.mask_[self.mask == 0] = -1
            Save(self.mask_, "mask", self.opt.is_sigmoid)
            Save(self.real_A, "DF", self.opt.is_sigmoid)
            Save(self.real_B, "OK", self.opt.is_sigmoid)
            Save(self.fake_B, "FIX_result", self.opt.is_sigmoid)
            print("save_mask~")
            # raise KeyError

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.opt.lambda_lossG * self.loss_G_GAN + self.opt.lambda_loss_L1 * self.loss_G_L1
        # TODO
        # self.loss_G = self.loss_G_GAN + lamda * self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
