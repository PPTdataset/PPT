'''
验证并计算最后得分
input:
pre_path: path for json of what we generate
val_path: path for json of ground truth
'''
import argparse
import json
import os
from os import listdir
import cv2
import numpy as np 
import random

def save_to_txt(pre_path, msg, mode='a'):
    path = pre_path + 'score.txt'  
    f = open(path, mode)
    f.write(msg + '\n') 
    f.close()

def string2pixel(pixel_str):
    pixel_str = pixel_str.split(",")
    pixel = [int(i) for i in pixel_str]
    return pixel

def dict2array(x_dict):
    x_array = np.zeros((128, 128), dtype=np.int)
    for c in x_dict["regions"]:
        for pixel_str in c["points"]:
            pixel = string2pixel(pixel_str)
            x_array[pixel[0], pixel[1]] = 1
    return x_array

def mIOU(array1, array2):
    assert array1.shape == array2.shape
    # inter.
    inter = (array1 & array2).sum()
    # union.
    union = (array1 | array2).sum()
    # iou.
    miou = inter / union
    return miou

def validate(pre_path, val_path):
    escape_num = 0
    overkill_num = 0
    miou_list = []
    save_to_txt(pre_path, "***********miou***********", 'w')
    for fn in os.listdir(val_path):
        if fn in os.listdir(pre_path):
            val_f = open(val_path + fn)
            pre_f = open(pre_path + fn) 
            val_dict = json.loads(val_f.read())
            pre_dict = json.loads(pre_f.read())

            val_array = dict2array(val_dict)
            pre_array = dict2array(pre_dict)       

            miou = mIOU(pre_array, val_array)
            miou_list.append(miou)
            print(miou)
            save_to_txt(pre_path, str(miou))
        else:
            escape_num += 1
    for fn in os.listdir(pre_path):
        if fn not in os.listdir(val_path) and fn[-4:] == 'json':
            overkill_num += 1
    # final_score = 0.5*(1-escape)+0.3*(1-overkill)+0.2*(mIOU)
    escape = escape_num / len(os.listdir(val_path))
    overkill = overkill_num / len(os.listdir(val_path))
    final_score = 0.5 * (1 - escape) + 0.3 * (1 - overkill) + 0.2 * sum(miou_list) / len(miou_list)
    
    print('escape = ', escape)
    print('overkill = ', overkill)
    print('final_score = ', final_score)
    save_to_txt(pre_path, "***********score***********")
    save_to_txt(pre_path, 'escape = ' + str(escape))
    save_to_txt(pre_path, 'overkill = ' + str(overkill))
    save_to_txt(pre_path, 'final_score = ' + str(final_score))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pre_path', required=True, help='path to image_real')
    parser.add_argument('--val_path', required=True, help='path to image_fake')
    # pre_path = './data_validation/focusight1_round1_train_part1/json_file/'
    # val_path = './data_validation/focusight1_round1_train_part1/json_file/'
    opt = parser.parse_args()

    validate(opt.pre_path, opt.val_path)

