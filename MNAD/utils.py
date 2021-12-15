import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn import metrics

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):

    return 10 * math.log10(1 / mse)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_img(img):

    img_re = copy.copy(img)
    
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    
    return img_re

def point_score(outputs, imgs):
    
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)
    normal = (1-torch.exp(-error))
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item()
    return score
    
def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def AUC(anomal_scores, labels):
    frame_auc = metrics.roc_auc_score(labels, anomal_scores)

    fpr, tpr, thresholds1 = metrics.roc_curve(labels, anomal_scores, pos_label=1)
    tpfp=(fpr,tpr)
    np.save("auc_s_NMAD.npy",tpfp)
    plt.plot(fpr,tpr)
    plt.savefig("auc_s_NMAD.png")
    tnr=1-fpr
    tp=tpr[np.nanargmin(np.absolute(tnr- tpr))]
    tn=tnr[np.nanargmin(np.absolute(tnr- tpr))]
    print(tp,tn,"close")
    tp=tpr[np.nanargmax(np.absolute(tnr+ tpr))]
    tn=tnr[np.nanargmax(np.absolute(tnr+ tpr))]
    print(tp,tn,"max")
    return frame_auc

def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
        
    return list_result

def tensor2im(input_image, imtype=np.uint8, is_sigmoid=False):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    assert input_image.dim() == 4
    # assert input_image.size(1) == 1
    n, c, h, w = input_image.size()

    input_image = input_image.permute(1, 0, 2, 3).reshape(1, c, n * h, w)
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        if is_sigmoid:
            image_numpy = np.clip(image_numpy, 0., 1.)
            # [0, 1] → [0, 255]
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
        else:
            image_numpy = np.clip(image_numpy, -1., 1.)
            # [-1, 1] → [0, 2] → [0, 1] → [0, 255]
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def diffimage(img1_, img2_):
    img1 = img1_.copy()
    img2 = img2_.copy()
    img1 = img1.astype(np.int)
    img2 = img2.astype(np.int)
    diff = abs(img1 - img2)
    return diff.astype(np.uint8)