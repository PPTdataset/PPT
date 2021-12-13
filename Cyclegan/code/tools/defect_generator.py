'''
缺陷生成器
input:
OK_img: cv2 file for OK_img
TC_img: cv2 file for TC_img

output:
DF_img: cv2 file for Defect Image
'''
import cv2
import numpy as np
import random
import json
from networkx.algorithms.operators.unary import reverse
import os
from os import listdir
import imgaug as ia
import imgaug.augmenters as iaa
sometimes = lambda aug: iaa.Sometimes(0.5, aug)  #建立lambda表达式
import queue
from collections import deque, Counter
import pyximport
pyximport.install()
from .utils import (Save_circles, Save_contours, Save, img2contours, contours2cont_max, is_inside_polygon, bgr2gray, gray2bgr, diffimage, Random, compare_defect, Clip)
# from .part1_rule_based import part1_rule_based_v3_1, part1_rule_based_v3_2, part1_rule_based_v4
# from .part2_rule_based import part2_rule_based_v1
# from .part2_generate_type import (part2_generate_type1_type2_v1, part2_generate_type1_v2, part2_generate_type1_v3, part2_generate_type2_v2, part2_generate_type2_v3,
                                #   part2_generate_type4_v1, part2_generate_type4_v2)
'''
A simple and common augmentation sequence
'''
ia.seed(1)
seq_tc = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.5)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(scale={
            "x": (0.8, 1.2),
            "y": (0.8, 1.2)
        }, translate_percent={
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2)
        }, rotate=(-180, 180), shear=(-8, 8))
    ],
    random_order=True)  # apply augmenters in random order

seq_tc2 = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # 对50%的图像进行镜像翻转
        iaa.Flipud(0.2),  # 对20%的图像做左右翻转
        sometimes(iaa.Crop(percent=(0, 0.1))),
        # 这里沿袭我们上面提到的sometimes，对随机的一部分图像做crop操作
        # crop的幅度为0到10%

        #对一部分图像做仿射变换
        sometimes(
            iaa.Affine(
                scale={
                    "x": (0.8, 1.2),
                    "y": (0.8, 1.2)
                },  # 图像缩放为80%到120%之间
                translate_percent={
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2)
                },  # 平移±20%之间
                rotate=(-45, 45),  # 旋转±45度之间
                shear=(-16, 16),  # 剪切变换±16度，（矩形变平行四边形）
                order=[0, 1],  # 使用最邻近差值或者双线性差值
                cval=(0, 255),  # 全白全黑填充
                mode=ia.ALL  # 定义填充图像外区域的方法
            )),

        # 使用下面的0个到5个之间的方法去增强图像。注意SomeOf的用法
        iaa.SomeOf(
            (0, 5),
            [
                # 将部分图像进行超像素的表示。
                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),

                #用高斯模糊，均值模糊，中值模糊中的一种增强。注意OneOf的用法
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                    iaa.MedianBlur(k=(3, 11)),
                ]),

                # 锐化处理
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                # 浮雕效果
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                # 边缘检测，将检测到的赋值0或者255然后叠在原图上
                sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                ])),

                # 加入高斯噪声
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),

                # 将1%到10%的像素设置为黑色
                # 或者将3%到15%的像素用原图大小2%到5%的黑色方块覆盖
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),

                # 5%的概率反转像素的强度，即原来的强度为v那么现在的就是255-v
                iaa.Invert(0.05, per_channel=True),

                # 每个像素随机加减-10到10之间的数
                iaa.Add((-10, 10), per_channel=0.5),

                # 像素乘上0.5或者1.5之间的数字.
                iaa.Multiply((0.5, 1.5), per_channel=0.5),

                # 将整个图像的对比度变为原来的一半或者二倍
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),

                # 将RGB变成灰度图然后乘alpha加在原图上
                iaa.Grayscale(alpha=(0.0, 1.0)),

                # 把像素移动到周围的地方。这个方法在mnist数据集增强中有见到
                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),

                # 扭曲图像的局部区域
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ],
            random_order=True  # 随机的顺序把这些操作用在图像上
        )
    ],
    random_order=True  # 随机的顺序把这些操作用在图像上
)

seq_ok = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Affine(scale={
            "x": (0.8, 1.2),
            "y": (0.8, 1.2)
        }, translate_percent={
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2)
        }, rotate=(-10, 10), shear=(-8, 8)),
        iaa.Crop(percent=(0, 0.1))  # random crops
    ],
    random_order=True)  # apply augmenters in random order

seq_4 = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.2))  # random crops
    ],
    random_order=True)  # apply augmenters in random order
'''
******************** direct_add ********************
'''


def direct_add_no_rule(OK_img, TC_img):
    '''
    从TC_img上裁剪随机矩形，直接替换进OK_img
    '''
    width, height = OK_img.shape[:2]

    # random crop from TC_img:
    crop_size_w = Random(10, width - 10)
    crop_size_h = Random(10, height - 10)
    # crop_size_w = Random(10, 50)
    # crop_size_h = Random(10, 50)

    pixel_w_tc = Random(0, width - crop_size_w)
    pixel_h_tc = Random(0, height - crop_size_h)
    pixel_w_df = Random(0, width - crop_size_w)
    pixel_h_df = Random(0, height - crop_size_h)

    DF_img = OK_img.copy()
    DF_img[pixel_w_df: pixel_w_df + crop_size_w, pixel_h_df: pixel_h_df + crop_size_h] = \
        TC_img[pixel_w_tc: pixel_w_tc + crop_size_w, pixel_h_tc: pixel_h_tc + crop_size_h]

    return gray2bgr(OK_img), gray2bgr(DF_img)


def direct_add(opt, OK_img, TC_img, roi_mask, data_augmentation, save_img, version="no_rule"):
    '''
    对TC_img使用Rule检测缺陷，直接将缺陷贴在OK_img上
    '''
    width, height = OK_img.shape[:2]
    TC_img_ori = TC_img.copy()
    if "1" in opt.data_augmentation:
        TC_img = seq_tc.augment_image(TC_img)
        seq_ok_det = seq_ok.to_deterministic()  # 确定一个数据增强的序列
        OK_img = seq_ok_det.augment_image(OK_img)
        roi_mask = seq_ok_det.augment_image(roi_mask)

    if version == "no_rule":
        return direct_add_no_rule(OK_img, TC_img), gray2bgr(roi_mask)

    # generate mask based on rule
    rule_based_mask = globals()[version](TC_img)
    is_defect = (rule_based_mask == 255)
    defect_list = np.nonzero(is_defect)
    DF_img = OK_img.copy()
    if len(defect_list[0]) == 0:
        return direct_add_no_rule(OK_img, TC_img), gray2bgr(roi_mask)
    else:
        DF_img[is_defect] = TC_img[is_defect]

    # difference between ok img and defect image
    if save_img:
        rand = Random(0, 999999999)
        crop_pixel = np.zeros((128, 128), )
        crop_pixel[is_defect] = 255

        OK_img = bgr2gray(OK_img)
        TC_img_ori = bgr2gray(TC_img_ori)
        TC_img = bgr2gray(TC_img)
        DF_img = bgr2gray(DF_img)

        compare = diffimage(OK_img, DF_img)

        boarder = 255 * np.ones((width, 10), )
        conjunction = np.concatenate([OK_img, boarder, TC_img_ori, \
                                        boarder, TC_img, boarder, DF_img, \
                                        boarder, crop_pixel, boarder, compare, \
                                        boarder, compare_defect(compare, 20), \
                                        boarder, compare_defect(compare, 30),
                                        boarder, compare_defect(compare, 40)], 1)
        cv2.imwrite("./saves/%s.bmp" % rand, conjunction)

    return gray2bgr(OK_img), gray2bgr(DF_img), gray2bgr(roi_mask)


'''
******************** seamless_clone ********************
'''


def seamless_clone_no_rule_v1(opt, OK_img, TC_img, roi_mask, TC_img_ori):
    '''
    从TC_img上裁剪随机矩形（正方形？），使用seamless_clone添加进OK_img
    '''
    width, height = OK_img.shape[:2]

    # random crop_size
    crop_size_h = random.randint(10, height - 10)
    crop_size_w = crop_size_h

    half_crop_size_h = int(crop_size_h / 2)
    half_crop_size_w = int(crop_size_w / 2)
    min_h = random.randint(0, width - crop_size_h)
    min_w = random.randint(0, height - crop_size_h)

    defect = TC_img[min_w:min_w + crop_size_w, min_h:min_h + crop_size_h]
    if "2" in opt.data_augmentation:
        # 对defect进行aug
        seq_tc_det = seq_tc.to_deterministic()  # 确定一个数据增强的序列
        defect = seq_tc_det.augment_image(defect)
    if "4" in opt.data_augmentation:
        defect = seq_4.augment_image(defect)

    # Create an all white mask
    mask = 255 * np.ones(defect.shape, defect.dtype)
    if 0:
        dst, contours = img2contours(OK_img, thr=50)
        if len(contours) > 0:
            # Save_contours(OK_img, contours, "contours")
            # cont = contours2cont_max(contours)
            # flag = 0
            # while flag != 1:
            #     center_h = random.randint(half_crop_size_h, height - half_crop_size_h)
            #     center_w = random.randint(half_crop_size_w, width - half_crop_size_w)
            #     flag = cv2.pointPolygonTest(cont, (center_h, center_w), False)
            inside_list = is_inside_polygon(contours)
            idx = Random(0, len(inside_list))
            # center = (inside_list[idx][1], inside_list[idx][0])
            center_w = Clip(inside_list[idx][1], half_crop_size_w, width - half_crop_size_w)
            center_h = Clip(inside_list[idx][0], half_crop_size_h, height - half_crop_size_h)
            center = (center_h, center_w)
        else:
            # The location of the center of the src in the dst
            center_h = random.randint(half_crop_size_h, height - half_crop_size_h)
            center_w = random.randint(half_crop_size_w, width - half_crop_size_w)
            center = (center_w, center_h)

        # Seamlessly clone src into dst and put the results in output
        # Save_circles(OK_img, center, "img_circle")
        gray = bgr2gray(OK_img)
        if gray[center] > 50:
            DF_img = cv2.seamlessClone(defect, OK_img, mask, center, cv2.MIXED_CLONE)
        else:
            DF_img = OK_img.copy()
    else:
        # The location of the center of the src in the dst
        center_h = random.randint(half_crop_size_h + 1, height - half_crop_size_h - 1)
        center_w = random.randint(half_crop_size_w + 1, width - half_crop_size_w - 1)
        center = (center_w, center_h)
        DF_img = cv2.seamlessClone(defect, OK_img, mask, center, cv2.MIXED_CLONE)

    if 0:
        rand = Random(0, 10)
        if rand == 0:
            DF_img = part2_generate_type1_type2_v1(OK_img)
        elif rand == 1:
            DF_img = part2_generate_type1_v3(OK_img)
        elif rand == 2:
            DF_img = part2_generate_type2_v3(OK_img)
        elif rand == 3:
            DF_img = part2_generate_type4_v2(OK_img)
        elif rand == 4:
            DF_img = part2_generate_type1_v2(OK_img)
        elif rand == 5:
            DF_img = part2_generate_type2_v2(OK_img)

    if "3" in opt.data_augmentation:
        seq_ok_det = seq_ok.to_deterministic()  # 确定一个数据增强的序列
        OK_img = seq_ok_det.augment_image(OK_img)
        DF_img = seq_ok_det.augment_image(DF_img)
    if "4" in opt.data_augmentation:
        seq_4_det = seq_4.to_deterministic()  # 确定一个数据增强的序列
        OK_img = seq_4_det.augment_image(OK_img)
        DF_img = seq_4_det.augment_image(DF_img)

    if opt.save_img:
        rand = Random(0, 999999999)
        crop_pixel = np.zeros((width, height), dtype="uint8")
        crop_pixel[min_w:min_w + crop_size_w, min_h:min_h + crop_size_h] = 255

        OK_img = bgr2gray(OK_img)
        TC_img_ori = bgr2gray(TC_img_ori)
        TC_img = bgr2gray(TC_img)
        DF_img = bgr2gray(DF_img)

        compare = diffimage(OK_img, DF_img)

        boarder = 255 * np.ones((width, 10), )
        conjunction = np.concatenate([OK_img, boarder, TC_img_ori, \
                                        boarder, TC_img, boarder, DF_img, \
                                        boarder, crop_pixel, boarder, compare, \
                                        boarder, compare_defect(compare, 20), \
                                        boarder, compare_defect(compare, 30), \
                                        boarder, compare_defect(compare, 40), \
                                        boarder, roi_mask], 1)
        cv2.imwrite("./saves/%s.bmp" % rand, conjunction)
        # Save(conjunction, "conjunction")

    return gray2bgr(OK_img), gray2bgr(DF_img), gray2bgr(roi_mask)


def seamless_clone_no_rule_v2(opt, OK_img, TC_img, roi_mask, TC_img_ori):
    '''
    从TC_img上裁剪随机矩形（正方形？），使用seamless_clone添加进OK_img
    '''
    width, height = OK_img.shape[:2]

    # random crop_size
    crop_size_h = Random(10, height - 10)
    crop_size_w = Random(10, width - 10)
    # crop_size_h = Random(10, 70)
    # crop_size_w = Random(10, 70)

    half_crop_size_h = int(crop_size_h / 2)
    half_crop_size_w = int(crop_size_w / 2)
    min_h = Random(0, width - crop_size_w)
    min_w = Random(0, height - crop_size_h)

    # Create an all white mask
    mask = np.zeros((width, height), OK_img.dtype)
    mask[min_w:min_w + crop_size_w, min_h:min_h + crop_size_h] = 255

    if "2" in opt.data_augmentation:
        # 对TC和mask进行aug
        seq_tc_det = seq_tc.to_deterministic()  # 确定一个数据增强的序列
        TC_img = seq_tc_det.augment_image(TC_img)
        mask = gray2bgr(mask)
        mask = seq_tc_det.augment_image(mask)
        mask = bgr2gray(mask)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask[5:15, 5:15] = 255

    # The location of the center of the src in the dst
    center = (int(width / 2), int(height / 2))

    # Seamlessly clone src into dst and put the results in output
    TC_img_s = TC_img[1:127, 1:127]
    mask_s = mask[1:127, 1:127]
    DF_img = cv2.seamlessClone(TC_img_s, OK_img, mask_s, center, cv2.MIXED_CLONE)

    rand = Random(0, 10)
    if rand == 0:
        DF_img = part2_generate_type1_type2_v1(OK_img)
    elif rand == 1:
        DF_img = part2_generate_type1_v3(OK_img)
    elif rand == 2:
        DF_img = part2_generate_type2_v3(OK_img)
    elif rand == 3:
        DF_img = part2_generate_type4_v2(OK_img)
    elif rand == 4:
        DF_img = part2_generate_type1_v2(OK_img)
    elif rand == 5:
        DF_img = part2_generate_type2_v2(OK_img)

    if "2" in opt.data_augmentation:
        seq_ok_det = seq_ok.to_deterministic()  # 确定一个数据增强的序列
        OK_img = seq_ok_det.augment_image(OK_img)
        DF_img = seq_ok_det.augment_image(DF_img)

    if opt.save_img:
        rand = Random(0, 999999999)
        crop_pixel = mask

        OK_img = bgr2gray(OK_img)
        TC_img_ori = bgr2gray(TC_img_ori)
        TC_img = bgr2gray(TC_img)
        DF_img = bgr2gray(DF_img)

        compare = diffimage(OK_img, DF_img)

        boarder = 255 * np.ones((width, 10), )
        conjunction = np.concatenate([OK_img, boarder, TC_img_ori, \
                                        boarder, TC_img, boarder, DF_img, \
                                        boarder, crop_pixel, boarder, compare, \
                                        boarder, compare_defect(compare, 20), \
                                        boarder, compare_defect(compare, 30), \
                                        boarder, compare_defect(compare, 40), \
                                        boarder, roi_mask], 1)
        cv2.imwrite("./saves/%s.bmp" % rand, conjunction)
        # Save(conjunction, "conjunction")

    return gray2bgr(OK_img), gray2bgr(DF_img), gray2bgr(roi_mask)


def seamless_clone(opt, OK_img, TC_img, roi_mask, mask_pro):
    seamless_clone_no_rule = globals()["seamless_clone_no_rule_v1"]
    '''
    从TC_img上裁剪随机矩形（正方形？），使用seamless_clone添加进OK_img
    '''
    width, height = OK_img.shape[:2]
    TC_img_ori = TC_img.copy()

    if opt.use_mask:
        # 处理二值化分割处理mask
        dst, contours = img2contours(OK_img, thr=200)
        if len(contours) > 0:
            cont = contours2cont_max(contours)
            for i in range(width):
                for j in range(height):
                    flag = cv2.pointPolygonTest(cont, (i, j), False)
                    if flag != -1:  # flag = -1 保留白色   flag == 1 保留黑色
                        roi_mask[j, i] = 0

    if "1" in opt.data_augmentation:
        seq_tc_det = seq_tc.to_deterministic()  # 确定一个数据增强的序列
        TC_img = seq_tc_det.augment_image(TC_img)

        seq_ok_det = seq_ok.to_deterministic()  # 确定一个数据增强的序列
        OK_img = seq_ok_det.augment_image(OK_img)
        roi_mask = seq_ok_det.augment_image(roi_mask)
        # Save(roi_mask, "roi_mask")
        # Save(OK_img, "OK_img")

    if "no_rule" in opt.version:
        seamless_clone_no_rule = globals()["seamless_clone_no_rule_%s" % opt.version[-2:]]
        return seamless_clone_no_rule(opt, OK_img, TC_img, roi_mask, TC_img_ori)

    if "rule_based" in opt.version:
        defect_mask = globals()[opt.version](TC_img)
    elif "self_iter" in opt.version:
        thr = opt.contrast_thr  # 达到一定的缺陷概率我们认为基本确定就是缺陷
        defect_mask = np.zeros((width, height), dtype="uint8")
        # rand = np.random.randint(0, 255, size=(width, height))
        # mask_pro = (mask_pro * 5).clip(max=255)
        # defect_mask[rand < mask_pro] = 255  # 缺陷概率越小，则该点被选择进行seamless clone的概率就越小
        defect_mask[thr < mask_pro] = 255
        # Save(mask_pro, "mask_pro")
        # Save(defect_mask, "defect_mask")
        if "2" in opt.data_augmentation:
            seq_tc_det = seq_tc.to_deterministic()  # 确定一个数据增强的序列
            TC_img = seq_tc_det.augment_image(TC_img)
            defect_mask = gray2bgr(defect_mask)
            defect_mask = seq_tc_det.augment_image(defect_mask)
            defect_mask = bgr2gray(defect_mask)
            retval, defect_mask = cv2.threshold(defect_mask, 127, 255, cv2.THRESH_BINARY)

    defect_list = np.nonzero(defect_mask == 255)
    if len(defect_list[0]) == 0:
        return seamless_clone_no_rule(opt, OK_img, TC_img, roi_mask, TC_img_ori)
    elif opt.defect_generator == 'seamless_clone' and 1:
        # TODO:获取概率最高的那个点，以它为中心随机生成一个矩形
        centers = np.reshape(np.where(mask_pro == np.max(mask_pro)), (-1, 2))
        index = Random(0, len(centers))
        center0 = Clip(centers[index], 6, width - 6)
        bond = min(center0[0], center0[1], width - center0[0], height - center0[1])

        # random crop_size
        if bond < 15:
            half_crop_size_h = Random(5, bond)
        elif bond > 60:
            half_crop_size_h = Random(50, bond)
        else:
            half_crop_size_h = Random(bond - 10, bond)
        half_crop_size_w = half_crop_size_h

        defect = TC_img[center0[0] - half_crop_size_w:center0[0] + half_crop_size_w, center0[1] - half_crop_size_h:center0[1] + half_crop_size_h]

        if "4" in opt.data_augmentation:
            defect = seq_4.augment_image(defect)

        # Create an all white mask
        mask = 255 * np.ones(defect.shape, defect.dtype)

        # The location of the center of the src in the dst
        center_h = random.randint(half_crop_size_h, height - half_crop_size_h)
        center_w = random.randint(half_crop_size_w, width - half_crop_size_w)
        center = (center_w, center_h)
        DF_img = cv2.seamlessClone(defect, OK_img, mask, center, cv2.MIXED_CLONE)

    elif opt.defect_generator == 'seamless_clone' and 0:
        # 获取外包络的矩形
        min_w = defect_list[0].min()
        max_w = defect_list[0].max()
        min_h = defect_list[1].min()
        max_h = defect_list[1].max()

        cs_w = max_w - min_w
        cs_h = max_h - min_h

        if min(cs_w, cs_h) <= 10:
            return seamless_clone_no_rule(opt, OK_img, TC_img, roi_mask, TC_img_ori)

        # Create mask
        mask = np.zeros((width, height), OK_img.dtype)
        mask[min_w:max_w, min_h:max_h] = 255

        # The location of the center of the src in the dst
        center = (int(width / 2), int(height / 2))

        # Seamlessly clone src into dst and put the results in output
        TC_img_s = TC_img[1:127, 1:127]

        mask_s = mask[1:127, 1:127]
        # mask_s = defect_mask[1: 127, 1:127]

        DF_img = cv2.seamlessClone(TC_img_s, OK_img, mask_s, center, cv2.MIXED_CLONE)
    elif opt.defect_generator == "direct_add":
        is_defect = (defect_mask == 255)
        mask = defect_mask
        DF_img = OK_img.copy()
        DF_img[is_defect] = TC_img[is_defect]
        # kernel_size = 3
        # DF_img = bgr2gray(DF_img)
        # blurred_img = cv2.GaussianBlur(DF_img, (kernel_size, kernel_size), 0)
        # DF_img = np.where(mask == 255, blurred_img, DF_img)

    if 0:
        rand = Random(0, 10)
        if rand == 0:
            DF_img = part2_generate_type1_type2_v1(OK_img)
        elif rand == 1:
            DF_img = part2_generate_type1_v3(OK_img)
        elif rand == 2:
            DF_img = part2_generate_type2_v3(OK_img)
        elif rand == 3:
            DF_img = part2_generate_type4_v2(OK_img)
        elif rand == 4:
            DF_img = part2_generate_type1_v2(OK_img)
        elif rand == 5:
            DF_img = part2_generate_type2_v2(OK_img)

    if "2" in opt.data_augmentation:
        seq_ok_det = seq_ok.to_deterministic()  # 确定一个数据增强的序列
        OK_img = seq_ok_det.augment_image(OK_img)
        DF_img = seq_ok_det.augment_image(DF_img)
    elif "4" in opt.data_augmentation:
        seq_4_det = seq_4.to_deterministic()  # 确定一个数据增强的序列
        OK_img = seq_4_det.augment_image(OK_img)
        DF_img = seq_4_det.augment_image(DF_img)

    # difference between ok img and defect image
    if opt.save_img:
        rand = Random(0, 999999999)
        # crop_pixel = mask
        crop_pixel = np.zeros((width, height), dtype="uint8")
        crop_pixel[center0[0] - half_crop_size_w:center0[0] + half_crop_size_w, center0[1] - half_crop_size_h:center0[1] + half_crop_size_h] = 255

        OK_img = bgr2gray(OK_img)
        TC_img_ori = bgr2gray(TC_img_ori)
        TC_img = bgr2gray(TC_img)
        DF_img = bgr2gray(DF_img)

        compare = diffimage(OK_img, DF_img)

        boarder = 255 * np.ones((width, 10), )
        conjunction = np.concatenate([OK_img, boarder, TC_img_ori, \
                                        boarder, TC_img, boarder, DF_img, \
                                        boarder, defect_mask, \
                                        boarder, crop_pixel, boarder, compare, \
                                        boarder, compare_defect(compare, 20), \
                                        boarder, compare_defect(compare, 30), \
                                        boarder, compare_defect(compare, 40), \
                                        boarder, roi_mask], 1)
        cv2.imwrite("./saves/%s.bmp" % rand, conjunction)
        # Save(conjunction, "conjunction")

    return gray2bgr(OK_img), gray2bgr(DF_img), gray2bgr(roi_mask)


'''
******************** ok2df_without_tc ********************
'''


def is_two_color(OK_img):
    width, height = OK_img.shape[:2]
    '''
    首先，要判断图像属于“黑白”还是“黑灰白”（含有灰色带状），type1和type2属于“黑白”图像上的缺陷，type3属于“黑灰白”上的，type4兼有。
    思路1：对OK_img全图统计灰度值出现频率，观察出现2个还是3个峰值，对2峰值情况应用type1，2；对3峰值应用type3。
    思路2：对OK_img使用合适的两个不同阈值二值化，再进行闭包检测，比较两个闭包内点相减的结果，若结果像素点数大于某个阈值(600个?)，则认为“黑白灰”
    '''
    '''
    思路1

    # 记录全图灰度值，取出现最多的前5个值
    boarder = []
    boarder.append(OK_img_gray[:, :])
    boarder = np.reshape(boarder, -1)

    # flag_array 记录峰值
    Counter_boarder = Counter(boarder)
    topk = min(5, len(Counter_boarder))
    flag_array = [Counter_boarder.most_common(topk)[i][0] for i in range(topk)]
    print(flag_array)
    '''
    '''
    思路2
    '''
    dst1, contours1 = img2contours(OK_img, thr=100)
    dst2, contours2 = img2contours(OK_img, thr=200)

    if len(contours1) == 0 or len(contours2) == 0:
        # 只能是第四类
        return True

    inside_list1 = is_inside_polygon(contours1)
    inside_list2 = is_inside_polygon(contours2)

    # 统计在inside_list1里，但是不在inside_list2里的点
    contrast_list = []
    contrast_mask = np.zeros((width, height), )
    for pixel1 in inside_list1:
        if pixel1 not in inside_list2:
            contrast_list.append(pixel1)
            contrast_mask[pixel1[1], pixel1[0]] = 255

    # contour_img1 = OK_img.copy()
    # contour_img2 = OK_img.copy()
    # contour_img = OK_img.copy()

    # contour_img1 = cv2.drawContours(contour_img1, contours1, -1, (0, 0, 255), 1)
    # contour_img2 = cv2.drawContours(contour_img2, contours2, -1, (0, 255, 0), 1)
    # contour_img = cv2.drawContours(contour_img, contours1, -1, (0, 0, 255), 1)
    # contour_img = cv2.drawContours(contour_img, contours2, -1, (0, 255, 0), 1)

    # rand = Random(0, 999999999)
    # cv2.imwrite("./saves/%d_contour1.bmp"% rand, contour_img1)
    # cv2.imwrite("./saves/%d_contour2.bmp"% rand, contour_img2)
    # cv2.imwrite("./saves/%d_contour.bmp"% rand, contour_img)
    # cv2.imwrite("./saves/%d_contrast_mask.bmp"% rand, contrast_mask)
    # print(len(contrast_list))
    return len(contrast_list) <= 600


def ok2df_without_tc(opt, OK_img, TC_img, roi_mask, mask_pro):
    '''
    对OK_img直接进行凸包生成，作为DF_img（不需要TC_img）
    '''
    width, height = OK_img.shape[:2]
    if "1" in opt.data_augmentation:
        TC_img = seq_tc.augment_image(TC_img)
        seq_ok_det = seq_ok.to_deterministic()  # 确定一个数据增强的序列
        OK_img = seq_ok_det.augment_image(OK_img)
        roi_mask = seq_ok_det.augment_image(roi_mask)

    # is_two = is_two_color(OK_img)
    '''
    应用 part2_generate_type
    '''
    if opt.version == "3302":
        rand = Random(0, 3)
        if rand == 0:
            DF_img = part2_generate_type1_v3(OK_img)
        elif rand == 1:
            DF_img = part2_generate_type2_v3(OK_img)
        elif rand == 2:
            DF_img = part2_generate_type4_v2(OK_img)

    elif opt.version == "2200":
        rand = Random(0, 2)
        if rand:
            DF_img = part2_generate_type1_v2(OK_img)
        else:
            DF_img = part2_generate_type2_v2(OK_img)

    elif opt.version == "mix_3302":
        rand = Random(0, 6)
        if rand == 0:
            DF_img = part2_generate_type1_type2_v1(OK_img)
        elif rand == 1:
            DF_img = part2_generate_type1_v3(OK_img)
        elif rand == 2:
            DF_img = part2_generate_type2_v3(OK_img)
        elif rand == 3:
            DF_img = part2_generate_type4_v2(OK_img)
        elif rand == 4:
            DF_img = part2_generate_type1_v2(OK_img)
        elif rand == 5:
            DF_img = part2_generate_type2_v2(OK_img)

    if "4" in opt.data_augmentation:
        seq_4_det = seq_4.to_deterministic()  # 确定一个数据增强的序列
        OK_img = seq_4_det.augment_image(OK_img)
        DF_img = seq_4_det.augment_image(DF_img)
    '''
    保存图片：difference between ok img and defect image
    '''
    if opt.save_img:
        # if save_img and not is_two:
        rand = Random(0, 999999999)

        OK_img = bgr2gray(OK_img)
        DF_img = bgr2gray(DF_img)

        compare = diffimage(OK_img, DF_img)

        boarder = 255 * np.ones((width, 10), )
        conjunction = np.concatenate([OK_img, boarder, DF_img, \
                                        boarder, compare, \
                                        boarder, compare_defect(compare, 20), \
                                        boarder, compare_defect(compare, 30),
                                        boarder, compare_defect(compare, 40)], 1)
        Save(conjunction, "%s_conjunction" % rand)
        Save(OK_img, "%s_OK_img" % rand)
        Save(roi_mask, "%s_roi_mask" % rand)
        print("image saved %d" % rand)

    return gray2bgr(OK_img), gray2bgr(DF_img), gray2bgr(roi_mask)


if __name__ == '__main__':
    saves_path = './saves/'
    try:
        os.makedirs(saves_path)
    except OSError:
        pass
    # OK_path = './data/focusight1_round1_train_part1/OK_Images/Image_87.bmp'
    # TC_path = './data/focusight1_round1_train_part1/TC_Images/42fWln6gGA0nl2O1h1UnhC6RlpMxVG.bmp'
    OK_path = './data/focusight1_round1_train_part2/OK_Images/'
    TC_path = './data/focusight1_round1_train_part2/TC_Images/0clSc38jtZ8ihhq3Pw6d5O4kTYt2aL.bmp'

    for fn in os.listdir(OK_path):
        OK_img = cv2.imread(OK_path + fn)
        TC_img = cv2.imread(TC_path)

        ok2df_without_tc(OK_img, TC_img, data_augmentation="0", save_img=1, version="no_rule")
