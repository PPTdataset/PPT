import argparse
import json
import os
import cv2
import numpy as np
import random
import shutil
from sklearn import metrics

def check_auc(results,labels):

    fpr1, tpr1, thresholds1 = metrics.roc_curve(labels, results, pos_label=255)  # (y, score, positive_label)
    fnr1 = 1 - tpr1
    eer_threshold1 = thresholds1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    EER1 = fpr1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    d_f1 = np.copy(results)
    d_f1[d_f1 >= eer_threshold1] = 255
    d_f1[d_f1 < eer_threshold1] = 0
    #f1_score = metrics.f1_score(labels, d_f1, pos_label=255)
    #print("AUC: {0}, EER: {1}, EER_thr: {2}, F1_score: {3}".format(metrics.auc(fpr1,tpr1), EER1,
    #                                                              eer_threshold1,f1_score))
    return metrics.auc(fpr1,tpr1)


def mIOU(array1, array2):
    # EPS = 1e-6
    assert array1.shape == array2.shape
    # inter.
    inter = (array1 & array2).sum()
    # union.
    union = (array1 | array2).sum()
    # iou.
    miou = float(inter) / float(union)
    # miou = (inter + EPS) / (union + EPS)
    return miou

if __name__ == '__main__':
    test_path="pre"
    gt_path="../TC_img_5k_gt"
    thr=10

    pre_path=test_path
    

    #dark=cv2.imread("tmp_thrimg/255.png")

    test_list=os.listdir(test_path)
    gt_list=os.listdir(gt_path)
    pre_list=os.listdir(pre_path)
    total_num=len(os.listdir(test_path))
    defect_num=0.0
    free_num=0.0
    escape_num=0.0 #fp
    overkill_num=0.0
    miou_list=[]
    auc_list=[]

    for fn in pre_list:
        img_pre=cv2.imread(os.path.join(pre_path,fn),cv2.IMREAD_GRAYSCALE)
        img_gt=cv2.imread(os.path.join(gt_path,fn),cv2.IMREAD_GRAYSCALE)
        img_test=cv2.imread(os.path.join(test_path,fn),cv2.IMREAD_GRAYSCALE)
        _,img_test=cv2.threshold(img_test,thr,255,cv2.THRESH_BINARY)
        if np.count_nonzero(img_gt)!=0 and np.count_nonzero(img_test)!=0:
            results=img_pre.flatten()
            labels=img_gt.flatten()
            
            auc=check_auc(results,labels)
            auc_list.append(auc)
            miou=mIOU(img_test,img_gt)
            miou_list.append(miou)
            print("{0} auc:{1} miou:{2}".format(fn,auc,miou))
        if np.count_nonzero(img_gt)==0:
            free_num+=1.0
            if np.count_nonzero(img_test)!=0:
                overkill_num+=1.0
                print(fn+" overkill")
        if np.count_nonzero(img_gt)!=0:
            defect_num+=1.0
            if np.count_nonzero(img_test)==0:
                escape_num+=1.0
                print(fn+" escape")

    escape = 1 - escape_num / defect_num  #分类为缺陷但是无缺陷——错误分类的无缺陷样本
    #正确分类的无缺陷样本
    overkill = 1 - overkill_num / free_num  
    #正确分类的缺陷样本
    miou = sum(miou_list) / len(miou_list)
    auc = sum(auc_list) / len(auc_list)
    print('escape = ', escape)
    print('overkill = ', overkill)
    print('miou = ', miou)
    print('auc=',auc)
