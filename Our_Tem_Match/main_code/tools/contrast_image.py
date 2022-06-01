'''
缺陷修复比较模型的test阶段，对修复前后图像进行对比的生成缺陷mask的算法
input: 
image_real, image_fake

output: 
mask_thr: a binary mask
mask_pro: a gray scale img where the posibility of defect increases along with the color from black to white 
'''
import argparse
# from skimage.feature import local_binary_pattern
# import skimage
import numpy as np
from PIL import Image
import cv2
from .utils import diffimage

def direct_contrast(image_real, image_fake, thr):
    width, height = image_real.shape[:2]
    
    mask_pro = diffimage(image_real, image_fake)

    is_defect = mask_pro > thr
    mask_thr = np.zeros((width, height),)
    mask_thr[is_defect] = 255
    
    return mask_thr.astype(np.uint8), mask_pro

# def LBP(image_real, image_fake, thr):
#     width, height = image_real.shape[:2]

#     # 参数设置
#     radius = 3	# LBP算法中范围半径的取值
#     n_points = 8 * radius # 领域像素点数

#     # LBP处理
#     lbp_real = local_binary_pattern(image_real, n_points, radius)
#     lbp_fake = local_binary_pattern(image_fake, n_points, radius)

#     mask_pro = diffimage(lbp_real, lbp_fake)
 
#     is_defect = mask_pro > thr
#     mask_thr = np.zeros((width, height),)
#     mask_thr[is_defect] = 255

#     return mask_thr, mask_pro

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_real_path', required=True, help='path to image_real')
    parser.add_argument('--image_fake_path', required=True, help='path to image_fake')
    parser.add_argument('--saves_path', required=True, help='path to saves')
    opt = parser.parse_args()

    image_real = cv2.imread(opt.image_real_path)
    image_fake = cv2.imread(opt.image_fake_path)
    mask_thr, mask_pro =  direct_contrast(image_real, image_fake)
    # mask_thr, mask_pro =  LBP(image_real, image_fake)

    
