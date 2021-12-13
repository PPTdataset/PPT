import torch
import torch.utils.data as data
import os
import cv2
import numpy as np
import shutil
from PIL import Image

gt_dir = 'Dataset/PPT/5k/ground_truth/sub/'
test_dir ='Dataset/PPT/5k/test/sub/'

for fn in os.listdir(gt_dir):
    img_gt=cv2.imread(os.path.join(gt_dir,fn),cv2.IMREAD_GRAYSCALE)
    img_test=cv2.imread(os.path.join(test_dir,fn[:-4])+'.bmp',cv2.IMREAD_GRAYSCALE)
    if np.count_nonzero(img_gt)==0:
        os.remove(os.path.join(gt_dir,fn))
        shutil.move(os.path.join(test_dir,fn[:-4])+'.bmp','Dataset/PPT/5k/test/good/'+fn[:-4]+'.bmp')

'''gt_dir = '../TC_img_5k_gt'
test_dir ='../TC_img_5k'

for fn in os.listdir(gt_dir):
    img_gt=cv2.imread(os.path.join(gt_dir,fn),cv2.IMREAD_GRAYSCALE)
    img_test=cv2.imread(os.path.join(test_dir,fn[:-4])+'.bmp',cv2.IMREAD_GRAYSCALE)
    if np.count_nonzero(img_gt)==0:
        Image.open(os.path.join(gt_dir,fn)).save(os.path.join(os.path.join('Dataset/PPT/5k/ground_truth/sub/',fn[:-4])+ '.png'))
        shutil.copy(os.path.join(test_dir,fn),'Dataset/PPT/5k/test/sub/'+fn)'''

