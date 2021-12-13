import cv2
import numpy as np 
import random
import math
cimport numpy as np
cimport cython
from .utils import (Clip, img2contours, contours2cont_max, is_inside_polygon, bgr2gray,
                    gray2bgr, Random)

cpdef copy_paste(img, img_ori, x, y, shift_range, thr_min=0, thr_max=255):
    dx = Random(-shift_range, shift_range)
    dy = Random(-shift_range, shift_range)
    width = img.shape[0]
    height = img.shape[1]
    if x + dx < 0 or x + dx >= width or y + dy < 0 or y + dy >= height:
        return img
    if int(img[x+dx, y+dy]) > thr_min and int(img[x+dx, y+dy]) < thr_max:
        return img
    shifted = cv2.warpAffine(img_ori, np.float32([[1, 0, dx], [0, 1, dy]]), \
                            (height, width))
    mask = np.zeros((width, height), dtype="uint8")
    rand_seed = Random(3, 5)
    cv2.circle(mask, (x+dy, y+dx), rand_seed, 255, -1)
    img = np.where(mask==255, shifted, img)

    return img

'''
******************** part2_generate_type1_type2_v1 ********************
'''
cpdef part2_generate_type1_type2_v1(OK_img):
    '''
    生成圆形    
    '''
    img = OK_img.copy()
    width, height = img.shape[:2]
    img = bgr2gray(img)

    dst, contours = img2contours(img, 100)

    if len(contours) == 0: #若没有轮廓，跳过
        return img

    cont = contours2cont_max(contours)
    boarder_pixels = np.squeeze(cont).reshape(-1, 2)

    # 对颜色为黑的边界,随机向外一圈晕染黑色
    for boarder_pixel in boarder_pixels:
        x = boarder_pixel[0]
        y = boarder_pixel[1]
        point_color = int(img[x, y])
        if x != 127 and x != 0 and y != 127 and y != 0:
            rand = Random(0, 40)
            if rand == 1:
                radius = Random(4, 10)
                cv2.circle(img, (x, y), radius, point_color, -1)

    # 画边界
    # contour_img = OK_img.copy()
    # contour_img = cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 1)
    # rand = Random(0, 999999999)
    # cv2.imwrite("./saves/%d_OK_img.bmp"% rand, OK_img)
    # cv2.imwrite("./saves/%d_contour.bmp"% rand, contour_img)
    # cv2.imwrite("./saves/%d_df_img.bmp"% rand, img)
    return img

'''
******************** part2_generate_type1_v2 ********************
'''
cpdef part2_generate_type1_v2(OK_img):
    '''
    边缘的白色凹缺陷
    生成多边形    
    '''
    img = OK_img.copy()
    cdef int width = img.shape[0]
    cdef int height = img.shape[1]
    img = bgr2gray(img)

    dst, contours = img2contours(img, 150)
    if len(contours) == 0: #若没有轮廓，跳过
        return img

    # 获取边缘检测结果
    cont = contours2cont_max(contours)
    boarder_pixels = np.squeeze(cont).reshape(-1, 2)

    # 去除边界点（即坐标x和y只要有一个是0或者127就不行）
    except_id = np.nonzero([127 in p or 0 in p for p in boarder_pixels])
    boarder_pixels = np.delete(boarder_pixels, except_id, axis=0)

    if len(boarder_pixels) == 0: #若没有轮廓，跳过
        return img
    # 随机选取边缘点，以其为中心，生成多边形？缺陷
    cdef int temp = 1
    cdef int x
    cdef int y
    cdef int i
    cdef int j
    cdef int loop_num
    cdef int area
    cdef int point_num
    cdef int radius
    cdef int flag

    mask = np.zeros((width, height), np.uint8)
    while temp > 0 and temp < 10000:
        temp += 1
        random_pixel_id = Random(0, len(boarder_pixels))
        x = boarder_pixels[random_pixel_id][0]
        y = boarder_pixels[random_pixel_id][1]
        point_color = int(img[x, y])
        if point_color > 210:
            radius = Random(8, 20)
            if temp % 2:
                cv2.circle(img, (x, y), radius, point_color, -1)
                cv2.circle(mask, (x, y), radius+2, 255, -1)
            else:
                area = 0
                loop_num = 0
                # print(loop_num)
                while area < 200 and loop_num < 200:
                    loop_num += 1
                    polygon = np.array([], dtype=int)
                    point_num = Random(4,10)
                    # for i in range(point_num):
                    #     polygon = np.append(polygon, [(x+Random(-20, 20)).clip(0, width), (y+Random(-20, 20)).clip(0, height)])
                    for i in range(point_num):
                        xita = Random(i*360/point_num, (i+1)*360/point_num)
                        radius = Random(10,40)
                        dx = int(radius * math.cos(math.radians(xita)))
                        dy = int(radius * math.sin(math.radians(xita)))
                        polygon = np.append(polygon, [Clip(x+dx, 0, width), Clip(y+dy, 0, height)])

                    polygon = np.reshape(polygon, (1, -1, 2))
                    im = np.zeros((width, height), dtype="uint8")
                    cv2.fillPoly(im, polygon, 255)

                    for i in range(width):
                        for j in range(height):
                            flag = cv2.pointPolygonTest(cont, (i, j), False)
                            if flag == 1:
                                im[j, i] = 0

                    area = np.sum(np.greater(im, 0))

                cv2.fillPoly(img, polygon, point_color)
                cv2.fillPoly(mask, polygon, 255)
                # print(loop_num)
            temp = 0

    cdef int kernel_size = 7
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    img = np.where(mask==255, blurred_img, img)

    # 画边界
    # print(temp)
    # contour_img = OK_img.copy()
    # contour_img = cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 1)
    # rand = Random(0, 999999999)
    # cv2.imwrite("./saves/%d_OK_img.bmp"% rand, OK_img)
    # cv2.imwrite("./saves/%d_contour.bmp"% rand, contour_img)
    # cv2.imwrite("./saves/%d_df_img.bmp"% rand, img)
    # cv2.imwrite("./saves/%d_blurred_img.bmp"% rand, blurred_img)

    return img

'''
******************** part2_generate_type2_v2 ********************
'''
cpdef part2_generate_type2_v2(OK_img):
    '''
    边缘的黑色凸缺陷
    生成多边形    
    '''
    img = OK_img.copy()
    cdef int width = img.shape[0]
    cdef int height = img.shape[1]
    img = bgr2gray(img)

    dst, contours = img2contours(img, 80)
    if len(contours) == 0: #若没有轮廓，跳过
        return img

    # 获取边缘检测结果
    cont = contours2cont_max(contours)
    boarder_pixels = np.squeeze(cont).reshape(-1, 2)

    # 去除边界点（即坐标x和y只要有一个是0或者127就不行）
    except_id = np.nonzero([127 in p or 0 in p for p in boarder_pixels])
    boarder_pixels = np.delete(boarder_pixels, except_id, axis=0)

    if len(boarder_pixels) == 0: #若没有轮廓，跳过
        return img
    # 随机选取边缘点，以其为中心，生成多边形？缺陷
    cdef int temp = 1
    cdef int x
    cdef int y
    cdef int i
    cdef int j
    cdef int loop_num
    cdef int area
    cdef int point_num
    cdef int radius
    cdef int flag

    temp = 1
    mask = np.zeros((width, height), np.uint8)
    while temp > 0 and temp < 10000:
        temp += 1
        random_pixel_id = Random(0, len(boarder_pixels))
        x = boarder_pixels[random_pixel_id][0]
        y = boarder_pixels[random_pixel_id][1]
        point_color = int(img[x, y])
        if point_color < 50:
            radius = Random(8, 20)
            if temp % 2:
                cv2.circle(img, (x, y), radius, point_color, -1)
                cv2.circle(mask, (x, y), radius+2, 255, -1)
            else:
                area = 0
                loop_num = 0
                # print(loop_num)
                while area < 200 and loop_num < 200:
                    loop_num += 1
                    polygon = np.array([], dtype=int)
                    point_num = Random(4,10)
                    # for i in range(point_num):
                    #     polygon = np.append(polygon, [(x+Random(-20, 20)).clip(0, width), (y+Random(-20, 20)).clip(0, height)])
                    for i in range(point_num):
                        xita = Random(i*360/point_num, (i+1)*360/point_num)
                        radius = Random(10,40)
                        dx = int(radius * math.cos(math.radians(xita)))
                        dy = int(radius * math.sin(math.radians(xita)))
                        polygon = np.append(polygon, [Clip(x+dx, 0, width), Clip(y+dy, 0, height)])

                    polygon = np.reshape(polygon, (1, -1, 2))
                    im = np.zeros((width, height), dtype="uint8")
                    cv2.fillPoly(im, polygon, 255)

                    for i in range(width):
                        for j in range(height):
                            flag = cv2.pointPolygonTest(cont, (i, j), False)
                            if flag == -1:
                                im[j, i] = 0

                    area = np.sum(np.greater(im, 0))

                cv2.fillPoly(img, polygon, point_color)
                cv2.fillPoly(mask, polygon, 255)
                # print(loop_num)
            temp = 0

    cdef int kernel_size = 7
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    img = np.where(mask==255, blurred_img, img)

    # 画边界
    # print(temp)
    # contour_img = OK_img.copy()
    # contour_img = cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 1)
    # rand = Random(0, 999999999)
    # cv2.imwrite("./saves/%d_OK_img.bmp"% rand, OK_img)
    # cv2.imwrite("./saves/%d_contour.bmp"% rand, contour_img)
    # cv2.imwrite("./saves/%d_df_img.bmp"% rand, img)
    # cv2.imwrite("./saves/%d_blurred_img.bmp"% rand, blurred_img)

    return img

'''
******************** part2_generate_type1_v3 ********************
'''
cpdef part2_generate_type1_v3(OK_img):
    '''
    边缘的黑色凸缺陷
    多个圆形的拷贝粘贴
    '''
    img = OK_img.copy()
    cdef int width = img.shape[0]
    cdef int height = img.shape[1]
    img = bgr2gray(img)

    dst, contours = img2contours(img, 100)
    if len(contours) == 0: #若没有轮廓，跳过
        return img

    # 获取边缘检测结果
    cont = contours2cont_max(contours)
    boarder_pixels = np.squeeze(cont).reshape(-1, 2)

    # 去除边界点（即坐标x和y只要有一个是0或者127就不行）
    except_id = np.nonzero([127 in p or 0 in p for p in boarder_pixels])
    boarder_pixels = np.delete(boarder_pixels, except_id, axis=0)

    if len(boarder_pixels) == 0: #若没有轮廓，跳过
        return img

    # 随机选取边缘点，以其为中心，向外复制粘贴
    temp = 0
    temp_point = 0
    img_ori = img.copy()
    while temp_point < 1 and temp < 10000:
        temp += 1
        random_pixel_id = Random(0, len(boarder_pixels))

        x = boarder_pixels[random_pixel_id][0]
        y = boarder_pixels[random_pixel_id][1]

        point_color = int(img[x, y])
        if point_color > 100:
            temp_point += 1
            for shift_range in range(1, 20):
                for _ in range(int(100 / shift_range)):
                    img = copy_paste(img, img_ori, x, y, shift_range, thr_min=180)

    cdef int kernel_size = 3
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    mask = np.zeros((width, height), np.uint8)
    cv2.circle(mask, (x, y), 30, 255, -1)
    img = np.where(mask ==255, blurred_img, img)

    # 画边界
    if 0:
        print(temp)
        contour_img = OK_img.copy()
        contour_img = cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 1)
        rand = Random(0, 999999999)
        # cv2.imwrite("./saves/%d_OK_img.bmp"% rand, OK_img)
        cv2.imwrite("./saves/%d_contour.bmp"% rand, contour_img)
        cv2.imwrite("./saves/%d_df_img.bmp"% rand, img)
        # cv2.imwrite("./saves/%d_shifted.bmp"% rand, shifted)

    return img

'''
******************** part2_generate_type2_v3 ********************
'''
cpdef part2_generate_type2_v3(OK_img):
    '''
    边缘的黑色凸缺陷
    多个圆形的拷贝粘贴
    '''
    img = OK_img.copy()
    cdef int width = img.shape[0]
    cdef int height = img.shape[1]
    img = bgr2gray(img)

    dst, contours = img2contours(img, 50)
    if len(contours) == 0: #若没有轮廓，跳过
        return img

    # 获取边缘检测结果
    cont = contours2cont_max(contours)
    boarder_pixels = np.squeeze(cont).reshape(-1, 2)

    # 去除边界点（即坐标x和y只要有一个是0或者127就不行）
    except_id = np.nonzero([127 in p or 0 in p for p in boarder_pixels])
    boarder_pixels = np.delete(boarder_pixels, except_id, axis=0)

    if len(boarder_pixels) == 0: #若没有轮廓，跳过
        return img

    # 随机选取边缘点，以其为中心，向外复制粘贴
    temp = 0
    temp_point = 0
    img_ori = img.copy()
    while temp_point < 1 and temp < 10000:
        temp += 1
        random_pixel_id = Random(0, len(boarder_pixels))

        x = boarder_pixels[random_pixel_id][0]
        y = boarder_pixels[random_pixel_id][1]

        point_color = int(img[x, y])
        if point_color < 50:
            temp_point += 1
            for shift_range in range(1, 20):
                for _ in range(int(100 / shift_range)):
                    img = copy_paste(img, img_ori, x, y, shift_range, thr_max=50)

    cdef int kernel_size = 3
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    mask = np.zeros((width, height), np.uint8)
    cv2.circle(mask, (x, y), 30, 255, -1)
    img = np.where(mask ==255, blurred_img, img)

    # 画边界
    if 0:
        print(temp)
        contour_img = OK_img.copy()
        contour_img = cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 1)
        rand = Random(0, 999999999)
        # cv2.imwrite("./saves/%d_OK_img.bmp"% rand, OK_img)
        # cv2.imwrite("./saves/%d_contour.bmp"% rand, contour_img)
        cv2.imwrite("./saves/%d_df_img.bmp"% rand, img)
        # cv2.imwrite("./saves/%d_shifted.bmp"% rand, shifted)

    return img

'''
******************** part2_generate_type4_v1 ********************
'''
cpdef part2_generate_type4_v1(OK_img):
    '''
    生成圆形
    '''
    img = OK_img.copy()
    width, height = img.shape[:2]
    gray = bgr2gray(img)

    # 二值化
    retval, dst = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    # 膨胀，白区域变大
    # dst = cv2.dilate(dst, None, iterations=1)
    # 腐蚀，白区域变小
    # dst = cv2.erode(dst, None, iterations=8)
    # 寻找图像中的轮廓
    contour_output = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = contour_output[-2:]
    if len(contours) == 0: #若没有轮廓，跳过
        return gray

    cont = contours2cont_max(contours)
    rand = Random(1, 2)    # 1对应缺陷4的第一种情况，2对应缺陷4的第二种情况  
    temp = 1
    while temp > 0 and temp < 10000:
        temp +=1
        tempx = Random(0, 127)
        tempy = Random(0, 127)
        radius = Random(4, 6)
        flag1 = cv2.pointPolygonTest(cont, (tempy, tempx), False)
        flag2 = abs(cv2.pointPolygonTest(cont, (tempy, tempx), True))
        # 随机的点不在边界上，且不在闭包内（即在黑色区域内）
        if tempx != 127 and tempx != 0 and tempy != 127 and tempy != 0 and flag1 == -1:
            # 对于缺陷4的第一种情况，flag2 > radius，第二种情况，flag2 == radius + 一个小的阈值
            if (rand == 1 and flag2 > radius) or (rand == 2 and flag2 == radius + 3):
                cv2.circle(gray, (tempy, tempx), radius, 255, -1)
                temp = 0

    # 画边界
    # print(temp)
    # contour_img = OK_img.copy()
    # contour_img = cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 1)
    # cv2.imwrite("./saves/c.bmp", contour_img)
    return gray

'''
******************** part2_generate_type4_v2 ********************
'''
cpdef part2_generate_type4_v2(OK_img):
    '''
    生成多个圆形
    '''
    img = OK_img.copy()
    width, height = img.shape[:2]
    img = bgr2gray(img)

    dst, contours = img2contours(img, 60)
    if len(contours) == 0: #若没有轮廓，跳过
        return img

    cont = contours2cont_max(contours)
    rand = Random(1, 3)    # 1对应缺陷4的第一种情况，2对应缺陷4的第二种情况  
    temp = 1
    while temp > 0 and temp < 10000:
        temp +=1
        x = Random(0, 127)
        y = Random(0, 127)
        radius = Random(4, 6)
        flag1 = cv2.pointPolygonTest(cont, (y, x), False)
        flag2 = abs(cv2.pointPolygonTest(cont, (y, x), True))
        # 随机的点不在边界上，且不在闭包内（即在黑色区域内）
        if x != 127 and x != 0 and y != 127 and y != 0 and flag1 == -1:
            # 对于缺陷4的第一种情况，flag2 > radius，第二种情况，flag2 == radius + 一个小的阈值
            if (rand == 1 and flag2 > radius) or (rand == 2 and flag2 == radius + 3):
                for _ in range(Random(5, 10)):
                    dx = Random(-8, 8)
                    dy = Random(-8, 8)
                    if x + dx < 0 or x + dx >= width or y + dy < 0 or y + dy >= height:
                        continue
                    cv2.circle(img, (y + dy, x + dx), Random(3, 6), 255, -1)
                temp = 0

    cdef int kernel_size = 5
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    mask = np.zeros((width, height), np.uint8)
    cv2.circle(mask, (y, x), 30, 255, -1)
    img = np.where(mask == 255, blurred_img, img)

    # 画边界
    # print(temp)
    # contour_img = OK_img.copy()
    # contour_img = cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 1)
    # cv2.imwrite("./saves/c.bmp", contour_img)
    return img