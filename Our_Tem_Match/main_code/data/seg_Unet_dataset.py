import os.path
import torch
from PIL import Image
import random
import cv2
import numpy as np
import time
import math
import pickle
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from tools.defect_generator import seamless_clone_defect_generator_plus, simple_clone_defect_generator
from tools.utils import Random, Save, FixCrop, get_transform_location, Combine_img, diffimage, gray2bgr, Make_dirs
from tools.contrast_image import direct_contrast
from options.test_options import TestOptions
from models import create_model
from util import util
from models.location_model import get_location_model, MeanModel, GroupLocationModel

EPS = 1e-10


def Clip(x, min_x, max_x):
    return max(min(x, max_x), min_x)


def check_valid_region(img):
    left = np.array((img.sum(0) != 0).nonzero())[0].min()
    right = np.array((img.sum(0) != 0).nonzero())[0].max() + 1
    top = np.array((img.sum(1) != 0).nonzero())[0].min()
    bottom = np.array((img.sum(1) != 0).nonzero())[0].max() + 1

    Mask = np.zeros(img.shape, img.dtype)
    Mask[top:bottom, left:right] = 255
    return Mask, [left, top, right, bottom]


class SegUnetDataset(BaseDataset):
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
        if opt.no_template: print("!!!!!!!!!!!!!! no mask !!!!!!!!!!!!!!")
        load_img_time = time.time()
        # create a path
        self.dir_OK = os.path.join(opt.dataroot, 'round_train', opt.part_name, 'OK_Images')
        self.dir_TC = os.path.join(opt.dataroot, 'round_test', opt.part_name, 'TC_Images')
        self.defect_generator=opt.defect_generator

        self.OK_paths = sorted(make_dataset(self.dir_OK))
        self.TC_paths = sorted(make_dataset(self.dir_TC))

        # get ok size
        self.OK_size = len(self.OK_paths)
        #self.OK_size = 30

        # get tc size
        self.TC_size = len(self.TC_paths)
        #self.TC_size = 100

        TC_img = cv2.imread(self.TC_paths[0])
        h_tc, w_tc = TC_img.shape[:2]

        # get transform
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc

        # preload OK_ori
        self.all_data_list = []
        self.grayscale = opt.input_nc == 1
        for index in range(self.OK_size):
            OK_img = Image.open(self.OK_paths[index]).convert('RGB')
            w_ok_ori, h_ok_ori = OK_img.size
            h_ok_ori_ds = int(h_ok_ori / opt.resize_ratio)
            w_ok_ori_ds = int(w_ok_ori / opt.resize_ratio)
            data_transform = get_transform_location(grayscale=self.grayscale, if_resize=True, resize_shape=[h_ok_ori_ds, w_ok_ori_ds])
            OK_img1 = data_transform(OK_img).unsqueeze(0)
            data_transform_ori = get_transform_location(grayscale=self.grayscale, if_resize=False)
            OK_img2 = data_transform_ori(OK_img).unsqueeze(0)
            self.all_data_list.append([OK_img1, OK_img2])

        print("load_img_time =", time.time() - load_img_time)
        print("topk =", opt.topk)
        print('resize_ratio =', opt.resize_ratio)

        # get gpu device
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        # TC_position, TC_position_score and TC_valid_keypoints
        # TC_property_dir = '%s/model/TC_property/' % (opt.ROOT)
        TC_property_dir = '%s/model/TC_property/%s' % (opt.ROOT, opt.name[6:])

        TC_property_name = '%s_r%s_c%s' % (opt.part_name, opt.resize_ratio, opt.input_nc)
        Make_dirs(TC_property_dir)
        save_dir = os.path.join(TC_property_dir, TC_property_name)

        # if os.path.exists(save_dir):
        #     f = open(save_dir, 'rb')
        #     self.TC_property = pickle.load(f)
        #     self.TC_position = self.TC_property['TC_position']
        #     self.TC_position_score = self.TC_property['TC_position_score']
        #     self.TC_valid_keypoints = self.TC_property['TC_valid_keypoints']
        #     f.close()
        #     print("############# load TC_position completed #############")
        # else:
        if 1:
            torch.cuda.synchronize()
            start_time = time.time()

            print("total TC_img num: %d" % self.TC_size)
            self.TC_position = np.zeros([self.OK_size, self.TC_size, 2], dtype='int')
            self.TC_position_score = np.zeros([self.OK_size, self.TC_size], dtype='float')
            self.TC_valid_keypoints = np.zeros([self.TC_size, 4], dtype='int')

            group_num = 200
            print('group_num =', group_num)
            tc_iter_total = math.ceil(self.TC_size / group_num)

            if False:#os.path.exists("../temp_data/TC_position/TC_position.pth"):
                self.TC_position=torch.load("../temp_data/TC_position/TC_position.pth")
                self.TC_position_score=torch.load("../temp_data/TC_position/TC_position_score.pth")
            else:
                for tc_iter in range(tc_iter_total):
                    torch.cuda.synchronize()
                    tc_time = time.time()
                    tc_index_list = np.arange(tc_iter * group_num, min((tc_iter + 1) * group_num, self.TC_size))
                    group_num_real = len(tc_index_list)

                    TC_img_list = []
                    iden_list = []
                    for tc_index in tc_index_list:
                        TC_img = cv2.imread(self.TC_paths[tc_index])
                        iden, keypoints = check_valid_region(TC_img)  # [128, 128]
                        self.TC_valid_keypoints[tc_index] = keypoints

                        iden = Image.fromarray(iden)
                        iden_list.append(iden)
                        TC_img = Image.fromarray(cv2.cvtColor(TC_img, cv2.COLOR_BGR2RGB))
                        TC_img_list.append(TC_img)

                    group_model1 = GroupLocationModel(opt, TC_img_list, iden_list, if_resize=True, ori_shape=[h_tc, w_tc])
                    group_model1.to(self.device)
                    group_model2 = GroupLocationModel(opt, TC_img_list, iden_list, if_resize=False, ori_shape=[h_tc, w_tc])
                    group_model2.to(self.device)

                    TC_img_norm_list = group_model2.TC_img_norm

                    for ok_iter in range(self.OK_size):
                        all_data = self.all_data_list[ok_iter]

                        # rough locating
                        # torch.cuda.synchronize()
                        # rough_time = time.time()
                        output = group_model1(all_data[0].to(self.device)).squeeze(0)
                        index = output.view(group_num_real, -1).argmax(1, keepdim=True)
                        position = torch.cat([index // output.shape[-1], index % output.shape[-1]], 1)
                        position = (position * self.opt.resize_ratio).cpu()  # [group_num_real, 2]
                        # torch.cuda.synchronize()
                        # print("rough_time: %.7f" %(time.time() - rough_time))

                        # accurate locating
                        # torch.cuda.synchronize()
                        # accurate_time = time.time()
                        swift = 30
                        b, c, h, w = all_data[1].shape  # 1, input_nc, h, w
                        data = all_data[1]

                        roi_data = []
                        for i in range(group_num_real):
                            data_single = data[:, :,
                                            Clip(position[i, 0] - swift, 0, h - h_tc - 2 * swift):Clip(position[i, 0] + swift + h_tc, h_tc + 2 * swift, h),
                                            Clip(position[i, 1] - swift, 0, w - w_tc - 2 * swift):Clip(position[i, 1] + swift + w_tc, w_tc + 2 * swift, w)]
                            roi_data.append(data_single)
                        data = torch.cat(roi_data, 1)

                        output = group_model2(data.to(self.device)).squeeze(0)
                        score, index = output.view(group_num_real, -1).max(1, keepdim=True)

                        position += torch.cat([index // output.shape[-1], index % output.shape[-1]], 1).cpu() - torch.tensor([swift, swift]).expand(group_num_real, -1)
                        position[:, 0] = torch.clamp(position[:, 0], 0, h - h_tc)
                        position[:, 1] = torch.clamp(position[:, 1], 0, w - w_tc)
                        self.TC_position[ok_iter, tc_index_list] = position.int()
                        self.TC_position_score[ok_iter, tc_index_list] = score.cpu().detach().squeeze().numpy() / np.array(TC_img_norm_list)
                        # torch.cuda.synchronize()
                        # print("accurate_time: %.7f" %(time.time() - accurate_time))
                        # print("accurate_time: %.7f" %(time.time() - rough_time))
                        # print("-----------------------------------------------------")

                    torch.cuda.synchronize()
                    print("finishing iters: %d/%d\ttime: %.7f" % (tc_iter, tc_iter_total, time.time() - tc_time))
                torch.save(self.TC_position,"../temp_data/TC_position/TC_position.pth")
                torch.save(self.TC_position_score,"../temp_data/TC_position/TC_position_score.pth")
            torch.cuda.synchronize()
            print("############# TC_position completed #############", time.time() - start_time)
            self.TC_property = {'TC_position': self.TC_position, 'TC_position_score': self.TC_position_score, 'TC_valid_keypoints': self.TC_valid_keypoints}

            TC_property = {'TC_position': self.TC_position, 'TC_position_score': self.TC_position_score, 'TC_valid_keypoints': self.TC_valid_keypoints}
            f = open(save_dir, 'wb')
            pickle.dump(TC_property, f, 0)
            f.close()

        check_num_ok = opt.topk + 1
        _, self.topk_indices = torch.tensor(self.TC_position_score).topk(check_num_ok, dim=0)

        self.not_gen_defect = 0
        self.gen_defect = 0

    def __getitem__(self, index):
        index_tc = index // self.opt.topk
        if self.opt.no_template:
            index_ok = self.topk_indices[index % self.opt.topk][index_tc]
        else:
            index_ok = self.topk_indices[index % self.opt.topk + 1][index_tc]

        index_mother = self.topk_indices[0][index_tc]

        # get OK_ori
        OK_img_ori = self.all_data_list[index_ok][1]
        OK_img_ori_mother = self.all_data_list[index_mother][1]

        # get TC_img_match
        if 1:
            TC_img_match = cv2.imread(self.TC_paths[index_tc])
        else:
            # debug for specific TC_img
            tc_name_list = ['79k78cUF1E78MRmc3KNWQvd6S2360T', '0cT5Z2IhG4ayhVT3I5NUV1rE2VsKF3', '4E1rutHW88wThtycfot9n41t2ngOug']
            self.tc_name_dict = {}
            tc_name = tc_name_list[random.randint(0, len(tc_name_list) - 1)]
            if tc_name in self.tc_name_dict:
                index_tc = self.tc_name_dict[tc_name]
            else:
                for i in range(self.TC_size):
                    if tc_name in self.TC_paths[i]:
                        index_tc = i
                        self.tc_name_dict[tc_name] = i
                        break
            TC_img_match = cv2.imread(os.path.join(self.dir_TC, '%s.bmp' % tc_name))
            index_ok = self.topk_indices[random.randint(1, self.opt.topk)][index_tc]
            index_mother = self.topk_indices[0][index_tc]
            OK_img_ori = self.all_data_list[index_ok][1]
            OK_img_ori_mother = self.all_data_list[index_mother][1]

        h_tc_ori, w_tc_ori = TC_img_match.shape[:2]

        # get TC_img
        use_same_tc = random.random() < self.opt.use_same_tc  # 0.66666
        if use_same_tc:
            index_tc_rand = index_tc
            TC_img = TC_img_match.copy()
        else:
            index_tc_rand = random.randint(0, self.TC_size - 1)
            TC_img = cv2.imread(self.TC_paths[index_tc_rand])

        # check valid region for TC_img_match, return mask for valid region and corresponding keypoints
        keypoints_tc_match = self.TC_valid_keypoints[index_tc]
        Mask = np.zeros(TC_img.shape, TC_img.dtype)
        Mask[keypoints_tc_match[1]:keypoints_tc_match[3], keypoints_tc_match[0]:keypoints_tc_match[2]] = 255

        # locate TC_img_match in OK_ori, get OK_img
        position = self.TC_position[index_ok][index_tc]
        OK_img = OK_img_ori[:, :, position[0]:position[0] + h_tc_ori, position[1]:position[1] + w_tc_ori]
        OK_img = util.tensor2im(OK_img, is_sigmoid=self.opt.is_sigmoid)  # cv2 form in rgb
        OK_img = cv2.cvtColor(OK_img, cv2.COLOR_BGR2RGB)  # cv2 form in bgr
        OK_img[Mask == 0] = 0

        # locate TC_img_match in mother_img, get Match_img
        position = self.TC_position[index_mother][index_tc]
        Match_img = OK_img_ori_mother[:, :, position[0]:position[0] + h_tc_ori, position[1]:position[1] + w_tc_ori]
        Match_img = util.tensor2im(Match_img, is_sigmoid=self.opt.is_sigmoid)  # cv2 form in rgb
        Match_img = cv2.cvtColor(Match_img, cv2.COLOR_BGR2RGB)  # cv2 form in bgr
        Match_img[Mask == 0] = 0
        
        defect_generator=globals()[self.defect_generator+'_defect_generator']
        # generate DF_img with TC_img and OK_img
        if random.random() >= self.opt.no_defect_rate:  # 0.4
            DF_img, show_mask = defect_generator(self.opt, OK_img.copy(), TC_img, use_same_tc, keypoints_tc_match)
        else:
            DF_img = OK_img.copy()
            show_mask = np.zeros(OK_img.shape, OK_img.dtype)

        # get Label
        Label = diffimage(OK_img, DF_img, self.grayscale)
        if not self.opt.mask_ignore:
            Label[Label > self.opt.contrast_thr] = 255
            Label[Label <= self.opt.contrast_thr] = 0
        else:
            ignore = (Label > self.opt.contrast_thr_low) * (Label <= self.opt.contrast_thr)
            Mask[ignore] = 0
            Label[Label > self.opt.contrast_thr] = 255
            Label[Label <= self.opt.contrast_thr_low] = 0

        if Label.sum() == 0:
            self.not_gen_defect += 1
        else:
            self.gen_defect += 1

        # save image
        if self.opt.save_img:
            image_list = [TC_img_match, OK_img, DF_img, Match_img, Label, Mask, show_mask]
            Save(Combine_img(self.opt.input_nc, image_list), random.randint(0, 99999999999), "%s/temp_data/saves/" % self.opt.ROOT)

        # to gray scale
        if self.input_nc == 1:
            OK_img = cv2.cvtColor(OK_img, cv2.COLOR_BGR2GRAY).reshape(h_tc_ori, w_tc_ori, 1)
            DF_img = cv2.cvtColor(DF_img, cv2.COLOR_BGR2GRAY).reshape(h_tc_ori, w_tc_ori, 1)
            Match_img = cv2.cvtColor(Match_img, cv2.COLOR_BGR2GRAY).reshape(h_tc_ori, w_tc_ori, 1)
        '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
        if self.opt.no_template:
            Match_img = Match_img * 0.
        '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''

        Label = Label.reshape(h_tc_ori, w_tc_ori, 1)
        Mask = cv2.cvtColor(Mask, cv2.COLOR_BGR2GRAY).reshape(h_tc_ori, w_tc_ori, 1)

        # get concat Input, channel=2
        # Input = np.concatenate([Match_img, DF_img], 2)
        Input = DF_img

        image_list2 = []
        for image in [OK_img, DF_img, Label, Mask, Input, Match_img]:
            image = image / 255.
            image = torch.from_numpy(image).float().permute(2, 0, 1)
            image_list2.append(image)

        OK_img, DF_img, Label, Mask, Input, Match_img = image_list2

        return {'OK_img': OK_img, 'DF_img': DF_img, 'Label': Label, 'Mask': Mask, 'Input': Input, 'Match_img': Match_img}

    def __len__(self):
        # Return the total number of images in the dataset
        return self.TC_size * self.opt.topk
