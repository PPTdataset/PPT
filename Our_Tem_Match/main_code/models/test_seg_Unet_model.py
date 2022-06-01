from .base_model import BaseModel
from . import networks
# from apex import amp
import torch
import os

class TestSegUnetModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestSegModel cannot be used during training time'
        return parser

    def __init__(self, opt):
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = []
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G' + opt.model_suffix]  # only generator is needed.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.is_sigmoid, opt.isTrain, opt)


        self.netG = self.netG.to(self.gpu_ids[0])
        # self.netG = amp.initialize(self.netG, opt_level=opt.opt_level)
        # self.netG = torch.nn.DataParallel(self.netG, self.gpu_ids)  # multi-GPUs
        setattr(self, 'netG' + opt.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input):
        # self.Match_img = input['Match_img']
        # self.Input = input['Input']
        # self.TC_img = input['TC_img']
        # self.Mask = input['Mask']
        # self.image_paths = input['TC_path']
        self.Match_img = input['Match_img'].to(self.device)
        self.Input = input['Input'].to(self.device)
        self.TC_img = input['TC_img'].to(self.device)
        self.Mask = input['Mask'].to(self.device)
        self.image_paths = input['TC_path']

    def forward(self):
        self.Pred_img = self.netG(self.Input)

    def optimize_parameters(self):
        """No optimization for test model."""
        pass

    def get_current_visuals(self):
        return {'TC_img': self.TC_img, 'Pred_img': self.Pred_img, 'Match_img': self.Match_img, 'Mask': self.Mask}

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