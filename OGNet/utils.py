# coding=utf-8
import torch
import cv2
import random
import numpy as np 
import os

def tensor2im(input_image, imtype=np.uint8, is_sigmoid=False):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if input_image.dim() == 3:
        input_image = input_image.unsqueeze(0)
    assert input_image.dim() == 4
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        if is_sigmoid:
            image_numpy = np.clip(image_numpy, 0., 1.)
            # [0, 1] → [0, 255]
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
        else:
            image_numpy = np.clip(image_numpy, -1., 1.)
            # [-1, 1] → [0, 2] → [0, 1] → [0, 255]
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def Make_dirs(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def Save_circles(img, center, name):
    img_circle = gray2bgr(img)
    img_circle = cv2.circle(img_circle, center, 2, (0, 0, 255), -1)
    cv2.imwrite("./saves/%s.bmp" %name, img_circle)

def Save_contours(img, contours, name):
    img_contour = gray2bgr(img)
    img_contour = cv2.drawContours(img_contour, contours, -1, (0, 0, 255), 1)
    cv2.imwrite("./saves/%s.bmp" %name, img_contour)

def Save(img, name):
    cv2.imwrite("./saves/%s.bmp" %name, img)

def img2contours(img, thr):
    img_gray = bgr2gray(img)
    # 二值化
    retval, dst = cv2.threshold(img_gray, thr, 255, cv2.THRESH_BINARY)
    # 边缘检测
    contour_output = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = contour_output[-2:]
    return dst, contours

def contours2cont_max(contours):
    assert len(contours) > 0
    cont_max_id = np.argmax([len(cont) for cont in contours])
    cont_max = contours[cont_max_id]
    return cont_max

def is_inside_polygon(contours):
    cont = contours2cont_max(contours)
    inside_list = []
    for i in range(128):
        for j in range(128):
            flag = cv2.pointPolygonTest(cont, (i, j), False)
            if flag == 1:
                inside_list.append([i, j])
    return np.array(inside_list).reshape(-1, 2)

def bgr2gray(img_):
    img = img_.copy()
    if len(img.shape) > 2:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img

def gray2bgr(img_):
    img = img_.copy()
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        return img

def diffimage(img1_, img2_):
    img1 = img1_.copy()
    img2 = img2_.copy()
    img1 = img1.astype(np.int)
    img2 = img2.astype(np.int)
    diff = abs(img1 - img2)
    return diff.astype(np.uint8)

def Random(x_, y_):
    x = int(x_)
    y = int(y_)-1
    return random.randint(x, y)

def compare_defect(compare, thr):
    mask = np.zeros((128, 128),)
    mask[compare > thr] = 255
    return mask

def Clip(x, min_x, max_x): 
    if type(x) == int :
        return max(min(x, max_x), min_x)
    elif x.size == 1:
        return max(min(x, max_x), min_x)
    else:
        return np.clip(x, min_x, max_x)

def Rotate(image):
    '''
    图像旋转
    '''
    #获取图像的尺寸
    #旋转中心
    angle = Random(0, 360)
    (h,w) = image.shape[:2]
    (cx,cy) = (w/2,h/2)
    
    #设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx,cy),-angle,1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    
    # 计算图像旋转后的新边界
    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))
    
    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0,2] += (nW/2) - cx
    M[1,2] += (nH/2) - cy
    
    return cv2.warpAffine(image,M,(nW,nH))

def Resize(image):
    '''
    ratio: 变换比例
    '''
    ratio = Random(1, 4)
    k = Random(0, 100) % 2
    if k == 0:
        ratio = 1 / ratio
    width = int(image.shape[0] * ratio)
    height = int(image.shape[1] * ratio)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

def Flip(image):
    '''
    direction = 1:  水平翻转
    direction = 0:  垂直翻转
    direction = -1: 水平加垂直翻转
    '''
    k = Random(0, 100) % 4
    if k == 0:
        direction = 1
    elif k == 1:
        direction = 0
    elif k == 2:
        direction = -1
    elif k == 3:
        return image
    return cv2.flip(image, direction)

def Affine(image):
    (h, w) = image.shape[:2]
    (ch, cw) = (h/2, w/2)

    matSrc = np.float32([[0, 0], [h - 1, 0], [0, w - 1]])              # 变换前的三个点坐标(左上角, 左下角,右上角)
    matDst = np.float32([[Random(0, ch), Random(0, cw)], [Random(ch + 1, h), Random(0, cw)], [Random(0, ch), Random(cw + 1, w)]])     # 变换后的三个点坐标(左上角, 左下角,右上角)
    
    matAffine = cv2.getAffineTransform(matSrc, matDst)      #mat 1 src 2 dst 形成组合矩阵
    return cv2.warpAffine(image, matAffine, (h, w))

def gaussian(ins, is_training, mean, stddev):
    if is_training:
        noise = ins.data.new(ins.size()).normal_(mean, stddev)
        noisy_image = ins + noise
        if noisy_image.max().data > 1 or noisy_image.min().data < -1:
            noisy_image = torch.clamp(noisy_image, -1, 1)
            if noisy_image.max().data > 1 or noisy_image.min().data < -1:
                raise Exception('input image with noise has values larger than 1 or smaller than -1')
        return noisy_image
    return ins