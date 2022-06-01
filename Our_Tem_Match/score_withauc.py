import argparse
from itertools import count
import json
from operator import pos
import os
import cv2
import numpy as np
import random
import shutil
from numpy.core.numeric import count_nonzero
from numpy.lib.function_base import append
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import label

def check_auc(results,labels):
    fpr1, tpr1, thresholds1 = metrics.roc_curve(labels, results, pos_label=255)  # (y, score, positive_label)
    #fnr1 = 1 - tpr1
    #eer_threshold1 = thresholds1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    #EER1 = fpr1[np.nanargmin(np.absolute((fnr1 - fpr1)))]
    #d_f1 = np.copy(results)
    #d_f1[d_f1 >= eer_threshold1] = 255
    #d_f1[d_f1 < eer_threshold1] = 0
    #f1_score = metrics.f1_score(labels, d_f1, pos_label=255)
    #print("AUC: {0}, EER: {1}, EER_thr: {2}, F1_score: {3}".format(metrics.auc(fpr1,tpr1), EER1,
    #                                                              eer_threshold1,f1_score))
    tpfp=(fpr1,tpr1)
    np.save("auc_s_%s.npy"%(name),tpfp)
    plt.plot(fpr1,tpr1)
    plt.savefig("auc_s_%s.png"%(name))
    return metrics.roc_auc_score(labels,results)


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

def auc_c(test_path,gt_path):

    tpfp=[]

    pre_list=os.listdir(test_path)
    thr_tmp=0
    step=1
    last=(0,0)
    thr_and_label=[]
    for fn in pre_list:
        img_pre=cv2.imread(os.path.join(test_path,fn),cv2.IMREAD_GRAYSCALE)
        img_gt=cv2.imread(os.path.join(gt_path,fn),cv2.IMREAD_GRAYSCALE)
        thr_tmp=0
        label=0
        if np.count_nonzero(img_gt)==0:
            label=0
        else:
            label=1
        thr_and_label.append((np.max(img_pre),label))
    for thr_tmp in range(0,255,1):
        pos_num=2868.0
        neg_num=2815.0
        tp_num=0.0
        fp_num=0.0
        for i in thr_and_label:
            if i[0]>thr_tmp and i[1]:
                pos_num+=1.0
            if i[0]>thr_tmp and ~i[1]:
                neg_num+=1.0
        now=(tp_num/pos_num,fp_num/neg_num)
        tpfp.append(now)
    '''while thr_tmp<=255:
        pos_num=0
        neg_num=0
        tp_num=0.0
        fp_num=0.0
        i=0
        for fn in pre_list:
            i+=1
            img_pre=cv2.imread(os.path.join(pre_path,fn),cv2.IMREAD_GRAYSCALE)
            img_gt=cv2.imread(os.path.join(gt_path,fn),cv2.IMREAD_GRAYSCALE)
            _,img_test=cv2.threshold(img_pre,thr_tmp,255,cv2.THRESH_BINARY)
            count_gt=np.count_nonzero(img_gt)
            count_test=np.count_nonzero(img_test)
            if count_gt!=0:
                pos_num+=1
                if count_test!=0:
                    tp_num+=1.0
            if count_gt==0:
                neg_num+=1
                if count_test!=0:
                    fp_num+=1.0
        now=(tp_num/pos_num,fp_num/neg_num)
        if abs(now[0]-last[0])<=0.01 or abs(now[1]-last[1])<=0.01:
            step=step+2
            thr_tmp+=step
        else:
            print(now)
            tpfp.append(now)
            step=1
            thr_tmp+=step
            last=now'''
    tpfp.sort(key=lambda x:x[1])
    np.save("auc_c_%s.npy"%(name),tpfp)
    tp=[]
    fp=[]
    lastfp=0.0
    auc=0.0
    for i in range(len(tpfp)):
        tp.append(tpfp[i][0])
        fp.append(tpfp[i][1])
        auc+=(fp[i]-lastfp)*tp[i]
        lastfp=fp[i]
    plt.plot(fp,tp)
    plt.savefig("auc_c_%s.png"%(name))
    return auc

def getthr(pre,gt):
    thr=0
    miou=0
    out_thr=0
    step=1
    while thr<255:
        miou_list=[]
        i=0
        for fn in os.listdir(gt):
            if i>50:
                #print(sum(miou_list) / len(miou_list),thr)
                break
            img_pre=cv2.imread(os.path.join(pre,fn),cv2.IMREAD_GRAYSCALE)
            img_gt=cv2.imread(os.path.join(gt,fn),cv2.IMREAD_GRAYSCALE)
            r,mask_thr=cv2.threshold(img_pre,thr,255,cv2.THRESH_BINARY)
            if np.count_nonzero(img_gt)!=0 and np.count_nonzero(mask_thr)!=0:
                miou=mIOU(mask_thr,img_gt)
                miou_list.append(miou)
                i+=1
        if miou<(sum(miou_list) / len(miou_list)):
            miou=sum(miou_list) / len(miou_list)
            print(thr,miou)
            out_thr=thr
            step=1
            thr+=step
        else:
            step+=2
            thr+=step
    return out_thr

def auc_s(test_path,gt_path,thr=None):
    pre_path=test_path
    test_list=os.listdir(test_path)
    gt_list=os.listdir(gt_path)
    pre_list=os.listdir(pre_path)
    total_num=len(os.listdir(test_path))

    results=[]
    labels=[]

    for i,fn in enumerate(pre_list):
        img_gt=cv2.imread(os.path.join(gt_path,fn),cv2.IMREAD_GRAYSCALE)
        img_test=cv2.imread(os.path.join(test_path,fn),cv2.IMREAD_GRAYSCALE)
        #_,img_test=cv2.threshold(img_test,thr,255,cv2.THRESH_BINARY)
        if np.count_nonzero(img_gt)!=0 and np.count_nonzero(img_test)!=0:
            results.append(img_test.flatten())
            labels.append(img_gt.flatten())
    results=np.asarray(results).flatten()
    labels=np.asarray(labels).flatten()
            
    mioulist=[]
    auclist=[]
    if thr!=None:
        for i,fn in enumerate(pre_list):
            img_gt=cv2.imread(os.path.join(gt_path,fn),cv2.IMREAD_GRAYSCALE)
            img_pre=cv2.imread(os.path.join(test_path,fn),cv2.IMREAD_GRAYSCALE)
            _,img_test=cv2.threshold(img_test,thr,255,cv2.THRESH_BINARY)
            if np.count_nonzero(img_gt)!=0 and np.count_nonzero(img_test)!=0:
                mioulist.append(mIOU(img_gt,img_test))
            if np.count_nonzero(img_gt)!=0 and np.count_nonzero(img_pre)!=0:
                auclist.append(check_auc(img_pre.flatten(),img_gt.flatten()))
        miou=sum(mioulist)/len(mioulist)
        auc=sum(auclist)/len(auclist)
    else:
        miou=0
        auc=check_auc(results,labels)
        

    #escape = 1 - escape_num / defect_num  #分类为缺陷但是无缺陷——错误分类的无缺陷样本
    #正确分类的无缺陷样本
    #overkill = 1 - overkill_num / free_num  
    #正确分类的缺陷样本
    
    return miou,auc


if __name__ == '__main__':
    test_path="../gan_img2img/temp_data/pre/cycle_gan" #cyclegan preimg now 10/11
    #test_path="pre_img_5k_Unet_simple"
    gt_path="../TC_img_5k_gt"
    name="Cyclegan"

    pre_path=test_path
    #aucc=auc_c(test_path,gt_path)
    #print("auc_c=:",aucc)
    
    #dark=cv2.imread("tmp_thrimg/255.png")
    #thr=getthr(test_path,gt_path)
    #print("thr=:",thr)

    miou,aucs=auc_s(test_path,gt_path,20)
    
    print('miou = ', miou)
    print('auc_s=',aucs)
