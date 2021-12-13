# coding=utf-8
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import numpy as np
from network import d_net, g_net
import torchvision.utils as vutils
from opts_fine_tune_discriminator import parse_opts_ft
from opts import parse_opts
#from fine_tune_dicriminator import fine_tune
from utils import gaussian
from dataloader import load_data
from sklearn import metrics
from utils import (Save_circles, Save_contours, Save, img2contours, contours2cont_max, is_inside_polygon, bgr2gray, gray2bgr, diffimage, Random, compare_defect, Clip ,tensor2im) 
import random
import cv2
import os
from dataloader import load_data_df
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seamless_clone_defect_generator_plus(thr, OK_img, TC_img, use_same_tc):
    #use_same_tc 概率 使用相同tc
    assert OK_img.shape == TC_img.shape
    show_mask = np.zeros(OK_img.shape, OK_img.dtype)
    DF_img_big = np.zeros(OK_img.shape, OK_img.dtype)

    # 得到去黑边的图
    l,t,r,b=0,0,128,128
    '''l, t, r, b = keypoints_tc_match
    if min(b - t, r - l) < 30:
        # TODO: 如果有效区域太少......
        return OK_img.copy(), np.zeros(OK_img.shape, OK_img.dtype)
    OK_img = OK_img[t:b, l:r]  #贴mask
    if use_same_tc:
        TC_img = TC_img[t:b, l:r]
        assert OK_img.shape == TC_img.shape'''
    height, width = OK_img.shape[:2]

    area = height * width
    if use_same_tc:
        min_crop = min(height, width) // 2
        max_area = area // 2
    else:
        min_crop = 20
        max_area = area // 4

    defect_area = random.randint(min_crop**2, max_area)

    defect_height = random.randint(min_crop, min(defect_area // min_crop, height))
    defect_width = min(defect_area // defect_height, width)

    top = random.randint(0, height - defect_height)
    left = random.randint(0, width - defect_width)

    defect = TC_img[top:top + defect_height, left:left + defect_width]

    if use_same_tc:
        OK_img_s = OK_img[top:top + defect_height, left:left + defect_width].copy().astype(np.int)

        if random.random() < 0.5:
            color = OK_img[height // 2, width // 2]
        else:
            color = OK_img_s[random.randint(0, defect_height - 1), random.randint(0, defect_width - 1)]

        OK_img_s = np.abs(OK_img_s - color)
        OK_img_s = np.mean(OK_img_s, axis=2)
        mask = np.zeros(defect.shape, defect.dtype)
        #low = OK_img_s < opt.region_thr
        low = OK_img_s < thr
        mask[low] = 255

        # 保底策略
        mask_h = random.randint(10, min(20, defect_height))
        mask_w = random.randint(10, min(20, defect_width))
        mask_top = random.randint(0, defect_height - mask_h)
        mask_left = random.randint(0, defect_width - mask_w)
        mask[mask_top:mask_top + mask_h, mask_left:mask_left + mask_w] = 255
    else:
        # Create an all white mask
        mask = 255 * np.ones(defect.shape, defect.dtype)

    # TODO：找到自己就是对齐的贴
    if use_same_tc:
        paste_center_h = top + (defect_height) // 2 - min(7, top)
        paste_center_w = left + (defect_width) // 2
    else:
        paste_center_h = random.randint(defect_height // 2, height - defect_height // 2)
        paste_center_w = random.randint(defect_width // 2, width - defect_width // 2)

    center = (paste_center_w, paste_center_h)

    try:
        # TODO NORMAL_CLONE or MIXED_CLONE
        # DF_img = cv2.seamlessClone(defect, OK_img, mask, center, cv2.NORMAL_CLONE)
        DF_img = cv2.seamlessClone(defect, OK_img, mask, center, cv2.MIXED_CLONE)
    except:
        Save(mask, '_mask')
        print(defect.shape)
        print(OK_img.shape)
        print(center)
        print(use_same_tc)

    if use_same_tc:
        show_mask[t + top:t + top + defect_height, l + left:l + left + defect_width] = mask
    else:
        show_mask[top:top + defect_height, left:left + defect_width] = mask

    Label = diffimage(OK_img, DF_img)
    Label[Label > thr] = 255
    Label[Label <= thr] = 0
    if Label.sum() > 0:
        # 把DF放回大图
        DF_img_big[t:b, l:r] = DF_img
        return DF_img_big, show_mask
    else:
        # 多边形
        if random.random() < 0.5:
            defect_h = random.randint(10, height // 3)
            defect_w = random.randint(10, width // 3)
            defect = np.zeros((defect_h, defect_w, 3), OK_img.dtype)
            polygon = np.array([[random.randrange(0, defect_h // 2), random.randrange(0, defect_w // 2)],
                                [random.randrange(0, defect_h // 2), random.randrange(defect_w // 2, defect_w)],
                                [random.randrange(defect_h // 2, defect_h), random.randrange(defect_w // 2, defect_w)],
                                [random.randrange(defect_h // 2, defect_h), random.randrange(0, defect_w // 2)]])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            defect = cv2.fillConvexPoly(defect, polygon, color)

            # 模糊瑕疵
            blur_kernel = random.randrange(7, 11, 2)
            defect = cv2.GaussianBlur(defect, (blur_kernel, blur_kernel), 0)
        # 折线段（可能只有一条线）
        else:
            defect_h = height
            defect_w = width
            defect = np.zeros((defect_h, defect_w, 3), OK_img.dtype)
            point_list = [(random.randrange(0, defect_h), random.randrange(0, defect_w)), (random.randrange(0, defect_h), random.randrange(0, defect_w)),
                          (random.randrange(0, defect_h), random.randrange(0, defect_w))]
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            defect = cv2.line(defect, point_list[0], point_list[1], color, random.randrange(1, 4))
            if random.randint(0, 1):
                defect = cv2.line(defect, point_list[1], point_list[2], color, random.randrange(1, 4))
            # 模糊瑕疵
            blur_kernel = random.randrange(3, 7, 2)
            defect = cv2.GaussianBlur(defect, (blur_kernel, blur_kernel), 0)

        # 瑕疵仿射变换，旋转等
        mat_rotate = cv2.getRotationMatrix2D(center=(defect.shape[0] * 0.5, defect.shape[1] * 0.5), angle=random.randrange(0, 360), scale=1)
        defect = cv2.warpAffine(defect, mat_rotate, (defect.shape[1], defect.shape[0]))
        mask = defect.copy()

        top = random.randint(0, height - defect_h)
        left = random.randint(0, width - defect_w)
        OK_img_s = OK_img[top:top + defect_h, left:left + defect_w].astype(np.int)
        defect = defect.astype(np.int)

        # 相加或相减，再截断到0-255之间，再赋值回原区域
        OK_img_s = OK_img_s - defect if random.random() > 0.5 else OK_img_s + defect
        OK_img_s = np.clip(OK_img_s, 0, 255).astype(np.uint8)
        OK_img[top:top + defect_h, left:left + defect_w] = OK_img_s
        DF_img_big[t:b, l:r] = OK_img

        show_mask[t + top:t + top + defect_h, l + left:l + left + defect_w] = mask
        return DF_img_big, show_mask

def make_DF_data(opt):
    data_path=opt.data_path+"/0/sub"
    df_path=opt.data_path+"/1/sub"
    listdata=os.listdir(data_path)
    for fn in listdata:
        Ori_img=cv2.imread(os.path.join(data_path,fn))

        use_same = random.random() < 0.66666  # 0.66666
        if use_same:
            TC_img=Ori_img
        else:
            index_rand = random.randint(0, len(listdata) - 1)
            TC_img = cv2.imread(os.path.join(data_path,listdata[index_rand]))
        
        # generate DF_img with TC_img and OK_img DF缺陷 生成缺陷部分
        #vutils.save_image(img,"ten.png",normalize=True)
        no_defect_rate =0.4
        if random.random() >= no_defect_rate:  # 0.4
            DF_img, show_mask = seamless_clone_defect_generator_plus(10, Ori_img.copy(), TC_img, use_same)
        else:
            DF_img = Ori_img.copy()
            show_mask = np.zeros(Ori_img.shape, Ori_img.dtype)
        cv2.imwrite(os.path.join(df_path,fn),DF_img)
        



def check_auc(g_model_path, d_model_path, i):
    opt_auc = parse_opts()
    opt_auc.batch_shuffle = False
    opt_auc.drop_last = False
    opt_auc.data_path = './data/test/'
    dataloader = load_data(opt_auc)
    model = OGNet(opt_auc, dataloader)
    model.cuda()
    d_results, labels = model.test_patches(g_model_path, d_model_path, i)
    d_results = np.concatenate(d_results)
    labels = np.concatenate(labels)
    fpr1, tpr1, thresholds1 = metrics.roc_curve(labels, d_results, pos_label=1)  # (y, score, positive_label)
    fnr1 = 1 - tpr1
    eer_threshold1 = thresholds1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    EER1 = fpr1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    d_f1 = np.copy(d_results)
    d_f1[d_f1 >= eer_threshold1] = 1
    d_f1[d_f1 < eer_threshold1] = 0
    f1_score = metrics.f1_score(labels, d_f1, pos_label=0)
    print("AUC: {0}, EER: {1}, EER_thr: {2}, F1_score: {3}".format(metrics.auc(fpr1,tpr1), EER1,
                                                                  eer_threshold1,f1_score))

class OGNet(nn.Module):
    @staticmethod
    def name():
        return 'Old is Gold: Redefining the adversarially learned one-class classification paradigm'

    def __init__(self, opt, dataloader):
        super(OGNet, self).__init__()
        self.adversarial_training_factor  = opt.adversarial_training_factor
        self.g_learning_rate = opt.g_learning_rate
        self.d_learning_rate = opt.d_learning_rate
        self.epoch = opt.epoch
        self.batch_size = opt.batch_size
        self.n_threads = opt.n_threads
        self.sigma_noise = opt.sigma_noise
        self.dataloader = dataloader
        self.g = g_net().cuda()
        self.d = d_net().cuda()
        self.image_grids_numbers = opt.image_grids_numbers
        self.filename = ''
        self.n_row_in_grid = opt.n_row_in_grid
        self.opt=opt

    def train(self, normal_class):
        self.g.train()
        self.d.train()

        # Set optimizators
        g_optim = optim.Adam(self.g.parameters(), lr=self.g_learning_rate)
        d_optim = optim.Adam(self.d.parameters(), lr=self.d_learning_rate)

        fake = torch.ones([self.batch_size], dtype=torch.float32).cuda()
        valid = torch.zeros([self.batch_size], dtype=torch.float32).cuda()
        print('Training until high epoch...')
        for num_epoch in range(self.epoch):
            # print("Epoch {0}".format(num_epoch))
            #make_DF_data(self.opt)
            ori_data=self.dataloader
            #df_data=load_data_df
            for i, data in enumerate(ori_data):
                input, gt_label = data
                input = input.cuda()

                g_optim.zero_grad()
                d_optim.zero_grad()
                sigma = self.sigma_noise ** 2
                input_w_noise = gaussian(input, 1, 0, sigma)  #Noise
                #input_w_noise=[]
                
                # Inference from generator
                g_output = self.g(input_w_noise)

                if i==1:
                    vutils.save_image(input[0:self.image_grids_numbers, :, :, :],
                                    './results/%03d_real_samples_epoch.png' % (num_epoch), nrow=self.n_row_in_grid, normalize=True)
                    vutils.save_image(g_output[0:self.image_grids_numbers, :, :, :],
                                    './results/%03d_fake_samples_epoch.png' % (num_epoch), nrow=self.n_row_in_grid, normalize=True)
                    vutils.save_image(input_w_noise[0:self.image_grids_numbers, :, :, :],
                                    './results/%03d_noise_samples_epoch.png' % (num_epoch), nrow=self.n_row_in_grid, normalize=True)

                ##############################################
                d_fake_output = self.d(g_output)
                d_real_output = self.d(input)
                d_fake_loss = F.binary_cross_entropy(torch.squeeze(d_fake_output), fake)
                d_real_loss = F.binary_cross_entropy(torch.squeeze(d_real_output), valid)
                d_sum_loss = 0.5 * (d_fake_loss + d_real_loss)
                d_sum_loss.backward(retain_graph=True)
                d_optim.step()
                g_optim.zero_grad()

                ##############################################
                g_recon_loss = F.mse_loss(g_output, input)
                g_adversarial_loss = F.binary_cross_entropy(d_fake_output.squeeze(), valid)
                g_sum_loss = (1-self.adversarial_training_factor)*g_recon_loss + self.adversarial_training_factor*g_adversarial_loss
                g_sum_loss.backward()
                g_optim.step()

                print('Epoch {0} / Iteration {1} '.format(num_epoch, i))
                print('g_loss {0}  d_loss {1}'.format(g_sum_loss,d_sum_loss))

                if i%10 == 0:
                    opts_ft = parse_opts_ft() #opts for phase two
                    

                    if num_epoch <= opts_ft.high_epoch:
                        g_model_name = 'g_low_epoch'
                        model_save_path = './models/' + g_model_name + "_" + str(num_epoch)
                        torch.save({
                            'epoch': num_epoch,
                            'g_model_state_dict': self.g.state_dict(),
                            'g_optimizer_state_dict': g_optim.state_dict(),
                        }, model_save_path)
                        
                        

                    if num_epoch >= opts_ft.high_epoch:
                        g_model_name = 'g_high_epoch'
                        d_model_name = 'd_high_epoch'
                        model_save_path = './models/' + g_model_name
                        torch.save({
                            'epoch': num_epoch,
                            'g_model_state_dict': self.g.state_dict(),
                            'g_optimizer_state_dict': g_optim.state_dict(),
                        }, model_save_path)

                        model_save_path = './models/' + d_model_name
                        torch.save({
                            'epoch': num_epoch,
                            'd_model_state_dict': self.d.state_dict(),
                            'd_optimizer_state_dict': d_optim.state_dict(),
                        }, model_save_path)

                        print('Epoch {0} / Iteration {1}: before phase two'.format(num_epoch, i))
                        high_epoch_g_model_name = 'g_high_epoch'
                        high_epoch_d_model_name = 'd_high_epoch'
                        g_model_save_path = './models/' + high_epoch_g_model_name
                        d_model_save_path = './models/' + high_epoch_d_model_name
                        check_auc(g_model_save_path, d_model_save_path,1)
                        #fine_tune() #Phase two   ###############s
                        print('After phase two: ')
                        high_epoch_g_model_name = 'g_high_epoch'
                        high_epoch_d_model_name = 'd_high_epoch'
                        g_model_save_path = './models/' + high_epoch_g_model_name
                        d_model_save_path = './models/' + high_epoch_d_model_name

                        check_auc(g_model_save_path, d_model_save_path,1)
                        return 0

    def test_patches(self,g_model_path, d_model_path,i):  #test all images/patches present inside a folder on given g and d models. Returns d score of each patch
        checkpoint_epoch_g = -1
        g_checkpoint = torch.load(g_model_path)
        self.g.load_state_dict(g_checkpoint['g_model_state_dict'])
        checkpoint_epoch_g = g_checkpoint['epoch']
        if checkpoint_epoch_g is -1:
            raise Exception("g_model not loaded")
        else:
            pass
        d_checkpoint = torch.load(d_model_path)
        self.d.load_state_dict(d_checkpoint['d_model_state_dict'])
        checkpoint_epoch_d = d_checkpoint['epoch']
        if checkpoint_epoch_g == checkpoint_epoch_d:
            pass
        else:
            raise Exception("d_model not loaded or model mismatch between g and d")

        self.g.eval()
        self.d.eval()
        labels = []
        d_results = []
        count = 0
        for input, label in self.dataloader:
            input = input.cuda()
            g_output = self.g(input)
            d_fake_output = self.d(g_output)
            count +=1
            d_results.append(d_fake_output.cpu().detach().numpy())
            labels.append(label)
        return d_results, labels

