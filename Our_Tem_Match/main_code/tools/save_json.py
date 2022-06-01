# -*- coding: utf-8 -*-
import numpy as np
import os
import ntpath
import json
import cv2
from numpy.core.fromnumeric import shape
from .utils import Combine_img, Save, compare_defect


def save_images_seg_v8(opt, visuals_all, img_path_all):
    for i in range(len(img_path_all)):
        img_path = img_path_all[i]
        visuals = {}
        for key in visuals_all:
            visuals[key] = visuals_all[key][i]

        short_path = ntpath.basename(img_path)
        name = os.path.splitext(short_path)[0]

        # 验证和出结果
        Pred_img = visuals['Pred_img'].squeeze()

        # 添加 Mask
        Mask = visuals['Mask'].squeeze()
        Pred_img = Pred_img * Mask

        point_list = (Pred_img > opt.sigmoid_thr).nonzero(as_tuple=False)  # [num, 2]
        height, width = Pred_img.shape

        # creat new dict
        new_dict = {'Height': {}, 'Width': {}, 'name': {}, 'regions': []}

        new_dict['Height'] = height
        new_dict['Width'] = width
        new_dict['name'] = "%s.bmp" % name

        new_points = {'points': []}
        for point in point_list:
            new_points['points'].append("%d, %d" % (point[0], point[1]))

        #保存到\temp_data\results_validation\的json
        if len(new_points['points']) > 0:
            new_dict['regions'].append(new_points)
            new_json = json.dumps(new_dict)  # dict2json
            save_dir = opt.val_js_path if opt.is_val else opt.results_dir
            f = open(save_dir + '%s.json' % name, 'w')
            f.write(new_json)
            f.close()

        #可视化保存
        if opt.save_img:
            Pred_img = (255 * visuals['Pred_img'].squeeze().cpu().numpy()).astype(np.uint8)
            Mask = (255 * visuals['Mask'].squeeze().cpu().numpy()).astype(np.uint8)
            match =(255 * visuals['Match_img'].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            if opt.input_nc == 1:
                TC_img = (255 * visuals['TC_img'].squeeze().cpu().numpy()).astype(np.uint8)
            else:
                TC_img = (255 * visuals['TC_img'].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)

            gt_path = os.path.join(opt.val_gt_path, "%s.bmp" % name)
            #draw_imgs = [TC_img, Pred_img, Mask, compare_defect(Pred_img, 127), compare_defect(Pred_img, 224)]
            com_path=os.path.join("../val_data/part3_compare","%s.bmp" % name)
            #print(gt_path,com_path)
            draw_imgs = [TC_img,match]
            
            if os.path.exists(com_path):
                com=cv2.imread(com_path)
                draw_imgs.append(com)

            if os.path.exists(gt_path):
                ground_truth = cv2.imread(gt_path)
                draw_imgs.append(ground_truth)
            '''draw_imgs.append(Pred_img)
            print("pre",Pred_img)
            draw_imgs.append(compare_defect(Pred_img, 127))'''
            
            conjunction = Combine_img(opt.input_nc, draw_imgs) #gry to rgb or rgb to gry
            Save(conjunction, name, "%s/temp_data/saves/%s" % (opt.ROOT, opt.part_name))
            #if opt.print_log:
            #   print("save~")

def save_images_seg_v9(opt, visuals_all, img_path_all):
    for i in range(len(img_path_all)):
        img_path = img_path_all[i]
        visuals = {}
        for key in visuals_all:
            visuals[key] = visuals_all[key][i]

        short_path = ntpath.basename(img_path)
        name = os.path.splitext(short_path)[0]

        # 验证和出结果
        Pred_img = visuals['Pred_img'].squeeze()

        # 添加 Mask
        Mask = visuals['Mask'].squeeze()
        Pred_img = Pred_img * Mask

        point_list = (Pred_img > opt.sigmoid_thr).nonzero(as_tuple=False)  # [num, 2]
        height, width = Pred_img.shape

        # creat new dict
        new_dict = {'version':{},'flags': {}, 'imagePath': {}, 'imageHeight': {}, 'imageWidth': {},'shape':[]}

        new_dict['version']="4.5.6"
        new_dict['imageHeight'] = height
        new_dict['imageWidth'] = width
        new_dict['imagePath'] = "..\TC_images\%s.bmp" % name
        new_points = {'points': []}#'label':{'0'},'group_id':{'null'},'shape_type':{'polygon'},'flags':{}}

        Pred_img = (255 * visuals['Pred_img'].squeeze().cpu().numpy()).astype(np.uint8)
        seg_img=compare_defect(Pred_img,127)
        con=cv2.findContours(seg_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contour=con[0]

        for p in contour:
            for pp in p:
                for point in pp:
                    new_points['points'].append("%d, %d" % (point[0], point[1]))

        #保存到\temp_data\results_validation\的json
        if len(new_points['points']) > 0:
            new_dict['shape'].append(new_points)
            new_json = json.dumps(new_dict)  # dict2json
            save_dir = os.path.join(opt.results_dir,opt.name)
            f = open(save_dir + '%s.json' % name, 'w')
            f.write(new_json)
            f.close()

        #可视化保存
        if opt.save_img:
            Pred_img = (255 * visuals['Pred_img'].squeeze().cpu().numpy()).astype(np.uint8)
            Mask = (255 * visuals['Mask'].squeeze().cpu().numpy()).astype(np.uint8)
            match =(255 * visuals['Match_img'].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            if opt.input_nc == 1:
                TC_img = (255 * visuals['TC_img'].squeeze().cpu().numpy()).astype(np.uint8)
            else:
                TC_img = (255 * visuals['TC_img'].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)

            gt_path = os.path.join(opt.val_gt_path, "%s.bmp" % name)
            #draw_imgs = [TC_img, Pred_img, Mask, compare_defect(Pred_img, 127), compare_defect(Pred_img, 224)]
            com_path=os.path.join("../val_data/part3_compare","%s.bmp" % name)
            tc_path=os.path.join("../val_data/part3_tc","%s.bmp"%name)
            #print(gt_path,com_path)
            seg_img=compare_defect(Pred_img,127)
            con=cv2.findContours(seg_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contour=con[0]
            label_img=TC_img.copy()
            cv2.drawContours(label_img,contour,-1,(0,255,0),1)

            if os.path.exists(gt_path):
                gt=cv2.imread(gt_path)
            else:
                gt=np.zeros((128,128,3))

            '''if os.path.exists(com_path):
                com=cv2.imread(com_path)
            else:
                com=np.zeros((128,128,3))'''
            draw_imgs = [TC_img,gt,label_img,match]
            
            conjunction = Combine_img(opt.input_nc, draw_imgs) #gry to rgb or rgb to gry
            if os.path.exists(com_path):
                Save(conjunction, name, "%s/temp_data/saves/%s" % (opt.ROOT, opt.name))
            if os.path.exists(tc_path):
                Save(Pred_img, name,"%s/temp_data/pre_img/%s" % (opt.ROOT, opt.name))
            #if opt.print_log:
            #   print("save~")


def save_images_recon(opt, visuals_all, img_path_all):
    for i in range(len(img_path_all)):
        img_path = img_path_all[i]
        visuals = {}
        for key in visuals_all:
            visuals[key] = visuals_all[key][i]

        short_path = ntpath.basename(img_path)
        name = os.path.splitext(short_path)[0]

        # 验证和出结果
        Pred_img = visuals['Pred_img'].squeeze()

        # 添加 Mask
        #Mask = visuals['Mask'].squeeze()
        #Pred_img = Pred_img * Mask
        
        #可视化保存
        if opt.save_img:
            Pred_img = (255 * visuals['Pred_img'].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            #Mask = (255 * visuals['Mask'].squeeze().cpu().numpy()).astype(np.uint8)
            #OK_img =(255 * visuals['_img'].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            Input = (255 * visuals['Input'].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            if opt.input_nc == 1:
                TC_img = (255 * visuals['TC_img'].squeeze().cpu().numpy()).astype(np.uint8)
            else:
                TC_img = (255 * visuals['TC_img'].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            pre_img=diffimage(Pred_img,TC_img,True)

            tc_path=os.path.join("../val_data/part3_tc","%s.bmp"%name)
            #draw_imgs = [TC_img, Pred_img, Mask, compare_defect(Pred_img, 127), compare_defect(Pred_img, 224)]
            #print(gt_path,com_path)
            seg_img=compare_defect(Pred_img,127)
            con=cv2.findContours(seg_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contour=con[0]
            label_img=TC_img.copy()
            cv2.drawContours(label_img,contour,-1,(0,255,0),1)
            
            draw_imgs = [TC_img,Input,label_img,Pred_img]
            
            conjunction = Combine_img(opt.input_nc, draw_imgs) #gry to rgb or rgb to gry
            if os.path.exists(tc_path):
                Save(pre_img, name,"%s/temp_data/pre_img/%s" % (opt.ROOT, opt.name))
                Save(conjunction, name, "%s/temp_data/saves/%s" % (opt.ROOT, opt.name))
            #if opt.print_log:
            #   print("save~")        