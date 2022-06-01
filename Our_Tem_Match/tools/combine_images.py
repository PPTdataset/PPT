'''
合并GAN修补前后的图像，进行对比
input: 
file name for img1 & img2

output: 
result
'''
import argparse
import torch
import torch.nn as nn
import torchvision
import math
import os
from os import listdir
from PIL import Image

def image_cat(fn1, fn2, image_path, saves_path):
    interval = 30
    img1 = Image.open(image_path + fn1)
    img2 = Image.open(image_path + fn2)

    # 单幅图像尺寸
    assert img1.size == img2.size
    width, height = img1.size

    # 创建空白长图
    result = Image.new(img1.mode, (width * 2 + interval, height), (255, 255, 255))

    # 拼接图片
    result.paste(img2, box=(0, 0))
    result.paste(img1, box=(width + interval, 0))

    # 保存图片
    result.save(saves_path + '%s.png' %fn1[:-9])

    print('save as')
    print(saves_path + '%s.png' %fn1[:-9])

    return -1
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_path', required=True, help='path to image')
    parser.add_argument('--saves_path', required=True, help='path to saves')
    opt = parser.parse_args()

    for fn1 in os.listdir(opt.image_path):
        for fn2 in os.listdir(opt.image_path):
            if fn1[:-8] == fn2[:-8] and fn1 != fn2:
                image_cat(fn1, fn2, opt.image_path, opt.saves_path)
