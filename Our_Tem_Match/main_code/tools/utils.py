import cv2
import random
import numpy as np 
import os
import torchvision.transforms as transforms
from PIL import Image


def get_transform_location(grayscale=False, if_resize=False, resize_shape=None):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if if_resize:
        assert resize_shape is not None
        transform_list.append(transforms.Resize(resize_shape, Image.BICUBIC))

    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)

def Combine_img(num_channel, img_list, data=None):
    assert len(img_list) > 1
    width, height = img_list[0].shape[:2]
    if num_channel == 1:
        boarder = 255 * np.ones((width, 10),)
        conjunction = bgr2gray(img_list[0])
    else:
        boarder = 255 * np.ones((width, 10, 3),)
        conjunction = gray2bgr(img_list[0])
    img_list.pop(0)
    i=0
    for img in img_list:
        if num_channel == 1:
            img = bgr2gray(img)
        else:
            img = gray2bgr(img)
        #print(i)
        i=i+1
        conjunction = np.concatenate([conjunction, boarder, img], 1)
    if data is not None:
        distance = width / len(data)
        for i in range(len(data)):
            cv2.putText(conjunction, '%.10f' %data[i], (20, 20 + int(distance * i)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    return conjunction

def FixCrop(img_, width, height):
    img = img_.copy()
    w, h = img.shape[:2]
    assert w >= width and h >= height
    w0 = random.randint(0, w - width)
    h0 = random.randint(0, h - height)
    crop = img[w0:w0+width, h0:h0+height]
    return crop

def Make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def Save_rectangle(img, left_top, right_bottom, name):
    img_rectangle = gray2bgr(img)
    img_rectangle = cv2.rectangle(img_rectangle, (left_top[0], left_top[1]), (right_bottom[0], right_bottom[1]), (0,0,255), 2)
    cv2.imwrite("./temp_data/saves/%s.bmp" %name, img_rectangle)

def Save_circles(img, center, name):
    img_circle = gray2bgr(img)
    img_circle = cv2.circle(img_circle, center, 2, (0, 0, 255), -1)
    cv2.imwrite("./temp_data/saves/%s.bmp" %name, img_circle)

def Save_contours(img, contours, name):
    img_contour = gray2bgr(img)
    img_contour = cv2.drawContours(img_contour, contours, -1, (0, 0, 255), 1)
    cv2.imwrite("./temp_data/saves/%s.bmp" %name, img_contour)

def Save(img, name, path="../temp_data/saves/"):
    Make_dirs(path)
    cv2.imwrite("%s/%s.bmp" %(path, name), img)

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

def diffimage(img1, img2, grayscale=True):
    if grayscale:
        img1 = bgr2gray(img1)
        img2 = bgr2gray(img2)
        img1 = img1.astype(np.int)
        img2 = img2.astype(np.int)
        diff = abs(img1 - img2)
        return diff.astype(np.uint8)
    else:
        img1 = img1.astype(np.int)
        img2 = img2.astype(np.int)
        diff = abs(img1 - img2)
        diff = diff.max(2)
        return diff.astype(np.uint8)

def real_diffimage(img1, img2, grayscale=True):
    if grayscale:
        img1 = bgr2gray(img1)
        img2 = bgr2gray(img2)
        img1 = img1.astype(np.int)
        img2 = img2.astype(np.int)
        diff = img1 - img2
        return diff
    else:
        img1 = img1.astype(np.int)
        img2 = img2.astype(np.int)
        diff = abs(img1 - img2)
        return diff.astype(np.uint8)

def Random(x_, y_):
    x = int(x_)
    y = int(y_)-1
    return random.randint(x, y)

def compare_defect(compare, thr):
    compare = bgr2gray(compare)
    mask = np.zeros((128, 128),)
    mask[compare > thr] = 255
    return mask.astype(np.uint8)

def Clip(x, min_x, max_x): 
    return max(min(x, max_x), min_x)

def Clip2(x, min_x, max_x): 
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

if __name__ == '__main__':
    saves_path = './saves/'
    try:
        os.makedirs(saves_path)
    except OSError:
        pass

    image_path = './data_validation/focusight1_round1_train_part2/TC_images/0j979AcFi4379QwB03u2N2Qk0ztTjL.bmp'
    image = cv2.imread(image_path)
    img_rotate = Rotate(image)
    cv2.imwrite(saves_path + "img_rotate.bmp", img_rotate)

    img_resize = Resize(image)
    cv2.imwrite(saves_path + "img_resize.bmp", img_resize)

    img_flip = Flip(image)
    cv2.imwrite(saves_path + "img_flip.bmp", img_flip)

    img_affine = Affine(image)
    cv2.imwrite(saves_path + "img_affine.bmp", img_affine)