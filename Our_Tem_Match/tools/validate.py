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
import shutil

from .json_generator import json_generator, json_generator_all

ESP = 1e-5


def save_to_txt(pre_path, msg, mode='a'):
    path = pre_path + '_score.txt'
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
    # EPS = 1e-6
    assert array1.shape == array2.shape
    # inter.
    inter = (array1 & array2).sum()
    # union.
    union = (array1 | array2).sum()
    # iou.
    miou = inter / union
    # miou = (inter + EPS) / (union + EPS)
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


def validate_v2(pre_path, val_path, ROOT, opt=None):
    #pre_path 训练结果
    #val 人工
    print("----------start validate--------------")
    val_json_path = val_path[:-1] + '_json/'

    #make json
    if not os.path.exists(val_json_path):
        os.makedirs(val_json_path)
        for fn in os.listdir(val_path):
            mask = cv2.imread(os.path.join(val_path, fn), 0)
            new_json = json_generator(mask, fn)
            if new_json is not None:
                f = open(val_json_path + '%s.json' % fn[:-4], 'w')
                f.write(new_json)
                f.close()

    IMGPATH = "%s/temp_data/saves/%s/" % (ROOT, val_path[-6:-1])
    escape_num = 0
    overkill_num = 0
    miou_list = []
    fn_dict = []
    # save_to_txt(pre_path, "***********miou***********", 'w')
    # print(os.listdir(val_path),os.listdir(val_json_path),os.listdir(pre_path))
    for fn in os.listdir(val_json_path):
        if fn in os.listdir(pre_path):
            val_f = open(val_json_path + fn)
            pre_f = open(pre_path + fn)
            val_dict = json.loads(val_f.read())
            pre_dict = json.loads(pre_f.read())

            val_array = dict2array(val_dict)
            pre_array = dict2array(pre_dict)

            miou = mIOU(pre_array, val_array)
            miou_list.append(miou)
            print(miou)
            if opt.save_img: fn_dict.append([IMGPATH + "%s.bmp" % fn[:-5], IMGPATH + "%.5f_%s.bmp" % (miou, fn[:-5])])
            # save_to_txt(pre_path, str(miou))
        elif fn in os.listdir(val_path):#首先要在5000张图之内
            if opt.save_img: fn_dict.append([IMGPATH + "%s.bmp" % fn[:-5], IMGPATH + "_escape_%s.bmp" % fn[:-5]])
            escape_num += 1
        #escapa 被标注为缺陷但是网络未识别 漏检率
    for fn in os.listdir(pre_path):
        if fn not in os.listdir(val_json_path) and fn[-4:] == 'json' and fn in os.listdir(val_path):
            if opt.save_img: fn_dict.append([IMGPATH + "%s.bmp" % fn[:-5], IMGPATH + "_overkill_%s.bmp" % fn[:-5]])
            overkill_num += 1
        #overkill 网络识别为缺陷但是标注不存在 过检率

    escape = 1 - escape_num / len(os.listdir(val_path))
    overkill = 1 - overkill_num / len(os.listdir(val_path))
    miou = sum(miou_list) / (len(miou_list) + ESP)#ESP = 1e-5
    final_score = 0.5 * escape + 0.3 * overkill + 0.2 * miou

    print('escape = ', escape)
    print('overkill = ', overkill)
    print('miou = ', miou)
    print("final_score = ",final_score)
    # save_to_txt(pre_path, "***********score***********")
    save_to_txt(pre_path, 'escape = ' + str(escape))
    save_to_txt(pre_path, 'overkill = ' + str(overkill))
    save_to_txt(pre_path, 'miou = ' + str(miou))
    save_to_txt(pre_path,"final_score = ",final_score)

    if opt.save_img:
        print("开始转换图片名")
        for ori_fn, new_fn in fn_dict:
            shutil.move(ori_fn, new_fn)


def validate_check(pre_path, val_path, ROOT, opt=None):
    val_json_path = val_path[:-1] + '_json/'

    if not os.path.exists(val_json_path):
        os.makedirs(val_json_path)
        for fn in os.listdir(val_path):
            mask = cv2.imread(os.path.join(val_path, fn), 0)
            new_json = json_generator(mask, fn)
            if new_json is not None:
                f = open(val_json_path + '%s.json' % fn[:-4], 'w')
                f.write(new_json)
                f.close()

    IMGPATH = "%s/temp_data/saves/%s/" % (ROOT, val_path[-6:-1])
    DATAROOT = os.path.join('%s/ok30tc1000/' % opt.ROOT, opt.part_name, 'TC_Images')
    escape_num = 0
    overkill_num = 0
    miou_list = []
    fn_dict = []
    i=0
    # save_to_txt(pre_path, "***********miou***********", 'w')
    for fn in os.listdir(val_json_path):
        if "%s.bmp" % fn[:-5] not in os.listdir(DATAROOT):
            continue
        if fn in os.listdir(pre_path):
            val_f = open(val_json_path + fn)
            pre_f = open(pre_path + fn)
            val_dict = json.loads(val_f.read())
            pre_dict = json.loads(pre_f.read())

            val_array = dict2array(val_dict)
            pre_array = dict2array(pre_dict)

            miou = mIOU(pre_array, val_array)
            miou_list.append(miou)
            if i%25==0:
                print(miou)
            i=i+1
            if opt.save_img: fn_dict.append([IMGPATH + "%s.bmp" % fn[:-5], IMGPATH + "%.5f_%s.bmp" % (miou, fn[:-5])])
            # save_to_txt(pre_path, str(miou))
        else:
            if opt.save_img: fn_dict.append([IMGPATH + "%s.bmp" % fn[:-5], IMGPATH + "_escape_%s.bmp" % fn[:-5]])
            escape_num += 1
    for fn in os.listdir(pre_path):
        if fn not in os.listdir(val_json_path) and fn[-4:] == 'json':
            if opt.save_img: fn_dict.append([IMGPATH + "%s.bmp" % fn[:-5], IMGPATH + "_overkill_%s.bmp" % fn[:-5]])
            overkill_num += 1
    # final_score = 0.5*(1-escape)+0.3*(1-overkill)+0.2*(mIOU)
    escape = 1 - escape_num / len(os.listdir(DATAROOT))
    overkill = 1 - overkill_num / len(os.listdir(DATAROOT))
    miou = sum(miou_list) / (len(miou_list) + ESP)

    print('escape = ', escape)
    print('overkill = ', overkill)
    print('miou = ', miou)
    # save_to_txt(pre_path, "***********score***********")
    save_to_txt(pre_path, 'escape = ' + str(escape))
    save_to_txt(pre_path, 'overkill = ' + str(overkill))
    save_to_txt(pre_path, 'miou = ' + str(miou))

    if opt.save_img:
        print("开始转换图片名")
        for ori_fn, new_fn in fn_dict:
            shutil.move(ori_fn, new_fn)


def validate_v2_forothers(pre_path, val_path):
    # for fn in os.listdir(val_path):
    #     mask = cv2.imread(os.path.join(val_path, fn), 0)
    #     new_json = json_generator(mask, fn)
    #     if new_json is not None:
    #         f = open(val_path[:-1] + '_json/%s.json' %fn[:-4], 'w')
    #         f.write(new_json)
    #         f.close()

    val_json_path = val_path[:-1] + '_json/'
    escape_num = 0
    overkill_num = 0
    miou_list = []
    save_to_txt(pre_path, "***********miou***********", 'w')
    for fn in os.listdir(val_json_path):
        if fn in os.listdir(pre_path):
            val_f = open(val_json_path + fn)
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
        if fn not in os.listdir(val_json_path) and fn[-4:] == 'json':
            overkill_num += 1
    # final_score = 0.5*(1-escape)+0.3*(1-overkill)+0.2*(mIOU)
    escape = 1 - escape_num / len(os.listdir(val_path))
    overkill = 1 - overkill_num / len(os.listdir(val_path))
    miou = sum(miou_list) / len(miou_list)

    print('escape = ', escape)
    print('overkill = ', overkill)
    print('miou = ', miou)
    save_to_txt(pre_path, "***********score***********")
    save_to_txt(pre_path, 'escape = ' + str(escape))
    save_to_txt(pre_path, 'overkill = ' + str(overkill))
    save_to_txt(pre_path, 'miou = ' + str(miou))


if __name__ == "__main__":

    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--pre_path', required=True, help='path to image_real')
    # parser.add_argument('--val_path', required=True, help='path to image_fake')
    # pre_path = ''
    # val_path = ''
    # opt = parser.parse_args()

    # validate(opt.pre_path, opt.val_path)
    validate_v2_forothers(\
        "/home/xiangli/YLFish/project_by_njust/temp_data/data/focusight1_round2_train_part1/TC_Images/",
        "/home/xiangli/YLFish/project/mask/part1/")
    validate_v2_forothers(\
        "/home/xiangli/YLFish/project_by_njust/temp_data/data/focusight1_round2_train_part2/TC_Images/",
        "/home/xiangli/YLFish/project/mask/part2/")
    validate_v2_forothers(\
        "/home/xiangli/YLFish/project_by_njust/temp_data/data/focusight1_round2_train_part3/TC_Images/",
        "/home/xiangli/YLFish/project/mask/part3/")
