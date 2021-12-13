'''
将搜集到的缺陷点按照是否连通分成不同的points，写入json文件
input: 
cv2 file for img
fn: file_name
'''
import cv2
import numpy as np 
import random
import json
from networkx.algorithms.operators.unary import reverse
import os
from os import listdir
import skimage.measure

# neighbor_hoods = 4, 8：[row, col]
OFFSETS_4 = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]
OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
             [-1,  0], [0,  0], [1,  0],
             [-1,  1], [0,  1], [1,  1]]

def reorganize(thr_num, new_dict, binary_img, num):
    rows, cols = binary_img.shape
    for i in range(1, num + 1):
        new_points =  {'points':[]}
        for row in range(rows):
            for col in range(cols):
                if binary_img[row][col] == i:
                    new_points['points'].append("%d, %d" % (row, col))
        if len(new_points['points']) > thr_num:
            new_dict['regions'].append(new_points)

    return new_dict

def neighbor_value(binary_img: np.array, offsets, reverse=False):
    rows, cols = binary_img.shape
    label_idx = 0
    rows_ = [0, rows, 1] if reverse == False else [rows-1, -1, -1]
    cols_ = [0, cols, 1] if reverse == False else [cols-1, -1, -1]
    for row in range(rows_[0], rows_[1], rows_[2]):
        for col in range(cols_[0], cols_[1], cols_[2]):
            label = 256
            if binary_img[row][col] < 0.5:
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row+offset[0]), rows-1)
                neighbor_col = min(max(0, col+offset[1]), cols-1)
                neighbor_val = binary_img[neighbor_row, neighbor_col]
                if neighbor_val < 0.5:
                    continue
                label = neighbor_val if neighbor_val < label else label
            if label == 255:
                label_idx += 1
                label = label_idx
            binary_img[row][col] = label
    return binary_img

# binary_img: bg-0, object-255; int
# neighbor_hoods = 4, 8
def Two_Pass(binary_img: np.array, neighbor_hoods):
    if neighbor_hoods == 4:
        offsets = OFFSETS_4
    elif neighbor_hoods == 8:
        offsets = OFFSETS_8
    else:
        raise ValueError

    binary_img = neighbor_value(binary_img, offsets, False)
    binary_img = neighbor_value(binary_img, offsets, True)

    return binary_img

def json_generator(img, fn, thr_num=0):
        width, height = img.shape

        # creat new dict
        new_dict = {'Height':{}, 'Width':{}, 'name':{}, 'regions':[]}

        new_dict['Height'] = height
        new_dict['Width'] = width
        new_dict['name'] = fn

        _, img = cv2.threshold(img, 127, 255, 0)
        
        # binary image: mask
        # binary_img = Two_Pass(img, 8)
        # num = int(binary_img.max())
        binary_img, num = skimage.measure.label(img, connectivity=2, background=0, return_num=True)
        new_dict = reorganize(thr_num, new_dict, binary_img, num)

        if new_dict['regions'] == []:
            return None
        else:
            new_json = json.dumps(new_dict)   # dict2json
            return new_json

if __name__ == "__main__":
    mask_path = './data_validation/focusight1_round1_train_part1/mask/'
    save_path = './data_validation/focusight1_round1_train_part1/json_file/'
    for fn in os.listdir(mask_path):
        img = cv2.imread(mask_path + fn, cv2.IMREAD_GRAYSCALE)
        json_generator(img, fn)