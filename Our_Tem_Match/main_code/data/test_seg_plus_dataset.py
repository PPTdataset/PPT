from PIL import Image
import cv2
import torch
import numpy as np
import time
import random
import os
import pickle
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from tools.utils import Random, Save, FixCrop, get_transform_location, Combine_img
from models.location_model import get_location_model
from util import util


def Clip(x, min_x, max_x):
    return max(min(x, max_x), min_x)


class TestSegPlusDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        dataset_time=time.time()

        BaseDataset.__init__(self, opt)
        if opt.no_template: print("!!!!!!!!!!!!!! no mask !!!!!!!!!!!!!!")
        # create a path
        self.dir_OK = os.path.join(opt.dataroot, 'Template')
        self.dir_TC = os.path.join(opt.dataroot, 'TC_img')
        # self.dir_TC = './debug/'

        # load images
        self.OK_paths = sorted(make_dataset(self.dir_OK))
        self.TC_paths = sorted(make_dataset(self.dir_TC))
        print("load image time",time.time()-dataset_time)
        dataset_time=time.time()

        # get ok size
        self.OK_size = len(self.OK_paths)
        #self.OK_size = 100

        # get tc size
        self.TC_size = len(self.TC_paths)
        #self.TC_size = 100

        print("OK and TC size=",self.OK_size,self.TC_size)

        # get transform
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc

        # preload OK_ori
        self.all_data_list = []
        self.grayscale = opt.input_nc == 1#是否是单通道灰度 否

        for index in range(self.OK_size):
            OK_img = Image.open(self.OK_paths[index]).convert('RGB')
            w_ok_ori, h_ok_ori = OK_img.size
            h_ok_ori_ds = int(h_ok_ori / opt.resize_ratio)
            w_ok_ori_ds = int(w_ok_ori / opt.resize_ratio)
            data_transform = get_transform_location(grayscale=self.grayscale, if_resize=True, resize_shape=[h_ok_ori_ds, w_ok_ori_ds])
            OK_img1 = data_transform(OK_img).unsqueeze(0)
            data_transform_ori = get_transform_location(grayscale=self.grayscale, if_resize=False)
            OK_img2 = data_transform_ori(OK_img).unsqueeze(0)
            self.all_data_list.append([OK_img1, OK_img2])#粗图和精度图

        print("pre OK_ori time", time.time() - dataset_time)
        dataset_time = time.time()

        # get transform
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc

        # get gpu id
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        if opt.TC_property is None: #如果注释了train部分
            # TC_property_dir = '%s/model/TC_property/' % (opt.ROOT)
            TC_property_dir = '%s/model/TC_property/%s' % (opt.ROOT, opt.name)

            TC_property_name = '%s_r%s' % (opt.name,opt.resize_ratio)
            save_dir = os.path.join(TC_property_dir, TC_property_name)
            f = open(save_dir, 'rb')
            opt.TC_property = pickle.load(f)
            f.close()

        self.TC_position = opt.TC_property['TC_position']
        self.TC_position_score = opt.TC_property['TC_position_score']
        self.TC_valid_keypoints = opt.TC_property['TC_valid_keypoints']
        print("load TC_property time", time.time() - dataset_time)

        _, self.topk_indices = torch.tensor(self.TC_position_score).topk(1, dim=0)

    def __getitem__(self, index_tc):
        # get tc
        TC_path = self.TC_paths[index_tc]
        TC_img = cv2.imread(TC_path)
        h_tc_ori, w_tc_ori = TC_img.shape[:2]

        # get mother img ori
        index_mother = self.topk_indices[0][index_tc]
        OK_img_ori_mother = self.all_data_list[index_mother][1]

        # check valid region for TC_img_match, return mask for valid region and corresponding keypoints
        keypoints_tc = self.TC_valid_keypoints[index_tc]
        Mask = np.zeros(TC_img.shape, TC_img.dtype)
        Mask[keypoints_tc[1]:keypoints_tc[3], keypoints_tc[0]:keypoints_tc[2]] = 255

        # locate TC_img in mother_img, get Match_img
        position = self.TC_position[index_mother][index_tc]
        Match_img = OK_img_ori_mother[:, :, position[0]:position[0] + h_tc_ori, position[1]:position[1] + w_tc_ori]
        Match_img = util.tensor2im(Match_img, is_sigmoid=self.opt.is_sigmoid)  # cv2 form
        Match_img = cv2.cvtColor(Match_img, cv2.COLOR_BGR2RGB)  # cv2 form in bgr
        Match_img[Mask == 0] = 0

        # to gray scale
        if self.input_nc == 1:
            TC_img = cv2.cvtColor(TC_img, cv2.COLOR_BGR2GRAY).reshape(h_tc_ori, w_tc_ori, 1)
            Match_img = cv2.cvtColor(Match_img, cv2.COLOR_BGR2GRAY).reshape(h_tc_ori, w_tc_ori, 1)
        Mask = cv2.cvtColor(Mask, cv2.COLOR_BGR2GRAY).reshape(h_tc_ori, w_tc_ori, 1)
        '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
        if self.opt.no_template:
            Match_img = Match_img * 0.
        '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''

        # get concat Input, channel=2
        #Input = np.concatenate([Match_img, TC_img], 2)
        if self.opt.input_nc==3:
            Input=TC_img
        elif self.opt.input_nc==6:
            Input = np.concatenate([Match_img, TC_img], 2)

        image_list2 = []
        for image in [Match_img, Input, TC_img, Mask]:
            image = image / 255.
            image = torch.from_numpy(image).float().permute(2, 0, 1)
            image_list2.append(image)
        Match_img, Input, TC_img, Mask = image_list2

        return {'Match_img': Match_img, 'Input': Input, 'TC_img': TC_img, 'TC_path': TC_path, 'Mask': Mask}

    def __len__(self):
        # Return the total number of images in the dataset
        return self.TC_size
