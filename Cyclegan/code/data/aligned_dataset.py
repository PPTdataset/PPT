import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import cv2
import numpy as np 
import sys
sys.path.append('%s/..'%sys.path[0])
from tools.defect_generator import direct_add, seamless_clone, ok2df_without_tc
from tools.utils import Random, Save
from tools.contrast_image import direct_contrast, LBP
from options.test_options import TestOptions
from models import create_model
from util import util
import torch
import torchvision.transforms as transforms


def get_transform_for_test(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    is_sigmoid = opt.is_sigmoid
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    osize = [256, 256]
    transform_list.append(transforms.Resize(osize, method))

    if convert:
        transform_list += [transforms.ToTensor()]
        if not is_sigmoid:
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def real_fake2pro(opt, visuals, image_path):
    # 在这里添加比较算法
    image_fake = util.tensor2im(visuals['fake'], is_sigmoid=opt.is_sigmoid)
    image_real = util.tensor2im(visuals['real'], is_sigmoid=opt.is_sigmoid)

    # 调整图像大小至128
    image_real = cv2.resize(image_real, (128, 128))
    image_fake = cv2.resize(image_fake, (128, 128))
    image_real = cv2.cvtColor(image_real, cv2.COLOR_BGR2GRAY)
    image_fake = cv2.cvtColor(image_fake, cv2.COLOR_BGR2GRAY)

    contrast_image = globals()["direct_contrast"]
    # contrast_image = globals()["LBP"]
    mask_thr, mask_pro = contrast_image(image_real, image_fake, thr=opt.contrast_thr)

    # Save(image_fake, "image_fake")
    # Save(image_real, "image_real")
    return mask_pro


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths

        self.dir_OK = os.path.join(opt.dataroot, 'round_train/%s/OK_Images/' %opt.part_name)  # create a path '/path/to/data/trainA'
        self.dir_TC = os.path.join(opt.dataroot, 'round_train/%s/TC_Images/' %opt.part_name)  # create a path '/path/to/data/trainB'
        #self.dir_NA = "/home/xiangli/YLFish/DefectDetection/data_CIFAR10/train/" #nature image？？

        self.OK_paths = sorted(make_dataset(self.dir_OK, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.TC_paths = sorted(make_dataset(self.dir_TC, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        #self.NA_paths = sorted(make_dataset(self.dir_NA, opt.max_dataset_size))

        self.OK_size = len(self.OK_paths)  # get the size of dataset A
        self.OK_size = 5000
        self.TC_size = len(self.TC_paths)  # get the size of dataset B
        print("OK and TC size----------------------")
        print(self.OK_size,self.TC_size)
        #self.NA_size = len(self.NA_paths)

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.defect_generator = globals()[opt.defect_generator]

        if "self_iter" in opt.version:
            ####################################
            ori_model = opt.model
            opt.model = 'test'

            ori_name = opt.name
            opt.name = opt.name_for_test

            ori_defect_generator = opt.defect_generator
            opt.defect_generator = opt.defect_generator_for_test

            ori_version = opt.version
            opt.version = opt.version_for_test

            opt.isTrain = False
            ####################################
            self.model_test = create_model(opt)      # create a model given opt.model and other options
            self.model_test.setup(opt)               # regular setup: load and print networks; create schedulers
            if opt.eval:
                self.model_test.eval()

            opt.model = ori_model
            opt.name = ori_name
            opt.defect_generator = ori_defect_generator
            opt.version = ori_version
            opt.isTrain = True


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        OK_path = self.OK_paths[index % self.OK_size]  # make sure index is within the range
        if self.opt.serial_batches:   # make sure index is within then range
            index_TC = index % self.TC_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_TC = random.randint(0, self.TC_size - 1)
        TC_path = self.TC_paths[index_TC]

        # random nattral image
        '''if self.opt.serial_batches:   # make sure index is within the range
            index_NA = index % self.NA_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_NA = random.randint(0, self.NA_size - 1)     
        NA_path = self.NA_paths[index_NA]'''
        
        # OK_img = Image.open(OK_path).convert('RGB')
        # TC_img = Image.open(TC_path).convert('RGB')

        OK_img = cv2.imread(OK_path)
        TC_img = cv2.imread(TC_path)
        #NA_img = cv2.imread(NA_path, 0)
        #NA_img = cv2.resize(NA_img, (128, 128))
        # Save(NA_img, "gray")

        '''if self.opt.if_NA:
            r = random.randint(0, 1)
            if r: 
                TC_img = NA_img'''

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, OK_img.shape[:2])
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        mask_transform = get_transform(self.opt, transform_params, grayscale=True)
        test_transform = get_transform_for_test(self.opt, transform_params, grayscale=True)

        if "self_iter" in self.opt.version:
            # 对图片进行test，再经过相同的数据增强变换得到mask，得到mask_pro
            TC_img_test = Image.open(TC_path).convert('RGB')
            TC_img_test = test_transform(TC_img_test)  # [1, 256, 256]
            TC_img_test = torch.unsqueeze(TC_img_test, 0)  # [1, 1, 256, 256]
            self.model_test.set_input({'A': TC_img_test, 'A_paths': TC_path})
            img_path = self.model_test.get_image_paths()     # get image paths
            self.model_test.test()           # run inference
            visuals = self.model_test.get_current_visuals()  # get image results
            mask_pro = real_fake2pro(self.opt, visuals, img_path)
            # Save(mask_pro, "mask_pro")
        else:
            mask_pro = None

        # create DF_img
        width, height = OK_img.shape[:2]
        mask = 255 * np.ones((width, height), OK_img.dtype)

        # 有一半的概率，不进行缺陷生成
        r = random.randint(0, 1)
        if r: 
            DF_img = OK_img.copy()
        else:
            OK_img, DF_img, mask = self.defect_generator(self.opt, OK_img, TC_img, mask, mask_pro)
            # OK_img, DF_img, mask = seamless_clone(self.opt, OK_img, TC_img, mask, mask_pro)

        if not self.opt.use_mask:
        # if Random(0, 2):
        # if 0:
            mask = 255 * np.ones((width, height, 3), OK_img.dtype)

        # OpenCV to PIL.Image
        OK_img = Image.fromarray(cv2.cvtColor(OK_img, cv2.COLOR_BGR2RGB))
        DF_img = Image.fromarray(cv2.cvtColor(DF_img, cv2.COLOR_BGR2RGB))
        mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

        OK_img = A_transform(OK_img)
        DF_img = B_transform(DF_img)
        mask = mask_transform(mask)

        return {'A': OK_img, 'B': DF_img, 'A_paths': OK_path, 'B_paths': OK_path, 'mask': mask}


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.OK_paths)
