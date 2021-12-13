import cv2
import queue
import numpy as np 
import random
import queue
import math
import time
from networkx.algorithms.operators.unary import reverse
from collections import deque, Counter
import skimage.measure
cimport numpy as np
cimport cython

'''
******************** v3.1 ********************
'''
cpdef part1_rule_based_v3_1(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cdef int width = img.shape[0]
    cdef int height = img.shape[1]

    # 记录边界灰度值，取出现最多的前5个值
    boarder = []
    
    boarder.append(img[0, :])
    boarder.append(img[127, :])
    boarder.append(img[:, 0])
    boarder.append(img[:, 127])
    boarder = np.reshape(boarder, -1)

    # flag_array = boarder
    Counter_boarder = Counter(boarder)
    cdef int topk = min(5, len(Counter_boarder))
    flag_array = [Counter_boarder.most_common(topk)[i][0] for i in range(topk)]
    mask = np.zeros((height, width),)

    cdef int thres = 2
    cdef int x
    cdef int y
    cdef int min_im_flag_abs
    for x in range(width):
        for y in range(height):
            # min_im_flag = min([int(img[x, y]) - int(flag) for flag in flag_array])
            min_im_flag_abs = min([abs(int(img[x, y]) - int(flag)) for flag in flag_array])
            # if min_im_flag > thres:
            if min_im_flag_abs > thres:
                mask[x, y] = 255
    '''
    消除边界为缺陷的误判
    '''
    # 二值化
    retval, dst = cv2.threshold(img, 24, 255, cv2.THRESH_BINARY)
    # cv2.imwrite("./saves/binary.bmp", dst)
    # 腐蚀和膨胀是对白色部分而言的，膨胀，白区域变大，最后的参数为迭代次数
    dst = cv2.dilate(dst, None, iterations=1)
    # 腐蚀，白区域变小
    dst = cv2.erode(dst, None, iterations=18)
    # 寻找图像中的轮廓
    # cv2.imwrite("./saves/processed_binary.bmp", dst)
    contour_output = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = contour_output[-2:]
    if len(contours) == 0: #若没有轮廓，跳过
        return mask
    cont = contours[0]

    # cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    # cv2.imwrite("./saves/contour.bmp", img)

    for x in range(width):
        for y in range(height):
            if cv2.pointPolygonTest(cont, (y, x), False) != 1:
                mask[x, y] = 0
    # cv2.imwrite("./saves/mask.bmp", mask)                    

    return mask

'''
******************** v3.2 ********************
'''
cpdef bfs(int x, int y, gray, is_defect_mat, flag_array, sign_mat, int n, int thres):
    direction = [[-1, 0], [-1, 1], [0, 1], [1, 1],
                [1, 0], [1, -1], [0, -1], [-1, -1]]
    q = queue.Queue()
    q.put((x, y))
    sign_mat[x][y] = 1
    while q.empty() == False:
        t = q.get()
        tx = t[0]
        ty = t[1]
        count = 0
        for j in range(len(flag_array)):  # 如果与周围一圈比大于阈值，则为缺陷部分
            if abs(int(gray[tx][ty]) - int(flag_array[j])) >= thres:
                count = count + 1
        if count == len(flag_array):
            is_defect_mat[tx][ty] = 1
            # print('缺陷位置', tx, ty)

        sign_mat[tx][ty] = 1

        for dr in direction:
            tempx = tx + dr[0]
            tempy = ty + dr[1]
            if tempx <= 0 or tempx >= n - 1 or tempy <= 0 or tempy >= n - 1 or sign_mat[tx][ty] == 1:
                continue
            if abs(int(gray[tx][ty] - int(gray[tempx][tempy]))) < thres:#如果与周围一圈像素差距小于阈值，则判定不是缺陷
                is_defect_mat[tx][ty] = 0
            q.put((tempx, tempy))

    return sign_mat, is_defect_mat

cpdef part1_rule_based_v3_2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cdef int width = gray.shape[0]
    cdef int height = gray.shape[1]
    cdef int n = width
    cdef int i = 0
    cdef int need = 0
    cdef int x
    cdef int y
    cdef int thres = 1

    flag_array = np.zeros(508, dtype=int)
    for x in range(width):
        for y in range(height):
            if x == 0 or y == 0 or x == 127 or y == 127:
                flag_array[i] = gray[x][y]
                #print("周围点坐标为：" + "(" + str(x) + "," + str(y) + ")" + "灰度值为：" + str(gray[x][y]))
                i = i + 1
                if gray[x][y] < 24:
                    need = need + 1

    #标记为1的是缺陷像素
    sign_mat = np.zeros((width, height))
    is_defect_mat = np.zeros((width, height))

    for x in range(width):
        for y in range(height):
            sign_mat, is_defect_mat = bfs(x, y, gray, is_defect_mat, flag_array, sign_mat, n, thres)

    if need != 0:
        # 二值化
        retval, dst = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        #cv2.imshow("binary",dst)
        # 腐蚀和膨胀是对白色部分而言的，膨胀，白区域变大，最后的参数为迭代次数
        dst = cv2.dilate(dst, None, iterations=1)
        # 腐蚀，白区域变小
        dst = cv2.erode(dst, None, iterations=18)
        # 寻找图像中的轮廓
        #cv2.imshow("processed binary",dst)
        contour_output = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = contour_output[-2:]
        if len(contours) > 0: #若没有轮廓，跳过
            cont = contours[0]
            gray = cv2.cvtColor(img, cv2.IMREAD_COLOR)
            cv2.drawContours(gray, contours, -1, (0, 0, 255), 2)
            #cv2.imshow('contour', gray)

            #cv2.waitKey(0)
            for x in range(width):
                for y in range(height):
                    flag = cv2.pointPolygonTest(cont,(y,x),False)
                    if flag != 1:
                        is_defect_mat[x][y] = 0

    is_defect = (is_defect_mat == 1)
    mask = np.zeros((height, width),)
    mask[is_defect] = 255
    return mask

'''
******************** v4 ********************
'''
cpdef BFS(int init_x, int init_y, flag_matrix, img, v, global_v, thres):
    '''
    BFS算法求解，广度优先搜索算法
    通常用队列（先进先出，FIFO）实现
        初始化队列Q；
        Q = {起点s}；标记s为已访问；
        while(Q非空):
            取Q队首元素u；u出队；
            if u == 目标状态 {...}
            所有与u相邻且未被访问的点进入队列；
            标记u为已访问；
    :return:
    '''
    cdef int width = img.shape[0]
    cdef int height = img.shape[1]
    dx = [-1, -1, 0, 1, 1, 1, 0, -1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]
    if not global_v:
        v = {(x, y): False for x in range(0, width) for y in range(0, height)}
    v[(init_x, init_y)] = True
    q = deque([(init_x, init_y, 0)]) # 前面两位指的是起点坐标，第三位指的是移动的步数
    while len(q) > 0:
        p = q.popleft()
        for i in range(8):
            x = p[0] + dx[i]
            y = p[1] + dy[i]
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            if v[(x, y)]:
                continue
            v[(x, y)] = True # 表示该点已经被访问

            # 若相邻两点的灰度值在阈值内，flag传递；若大于阈值, 停止渲染
            if abs(int(img[x, y]) - int(img[p[0], p[1]])) < thres:
                flag_matrix[x, y] = flag_matrix[p[0], p[1]]
                q.append((x, y, p[2] + 1))

    return flag_matrix

cpdef neighbor_flag(int x, int y, neighbor_flag_list, flag_matrix):
    cdef int width = flag_matrix.shape[0]
    cdef int height = flag_matrix.shape[1]
    # 邻域8个点查看flag，记录
    dx = [-1, -1, 0, 1, 1, 1, 0, -1]
    dy = [0, 1, 1, 1, 0, -1, -1, -1]
    for i in range(8):
        nx = x + dx[i]
        ny = y + dy[i]
        if nx < 0 or nx >= width or ny < 0 or ny >= height:
            continue
        if flag_matrix[nx, ny] == 1 or flag_matrix[nx, ny] == 2:
            neighbor_flag_list.append(flag_matrix[nx, ny])
    return neighbor_flag_list

cpdef part1_rule_based_v4(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cdef int width = img.shape[0]
    cdef int height = img.shape[1]

    cdef int thres = 4
    cdef int thr_num = 3
    cdef int global_v = 1  # 全局队列（global_v = True）的用时少

    # 记录边界灰度值，取出现最多的前5个值
    boarder = []
    
    boarder.append(img[0, :])
    boarder.append(img[127, :])
    boarder.append(img[:, 0])
    boarder.append(img[:, 127])
    boarder = np.reshape(boarder, -1)

    # flag_array = boarder
    Counter_boarder = Counter(boarder)
    topk = min(5, len(Counter_boarder))
    flag_array = [Counter_boarder.most_common(topk)[i][0] for i in range(topk)]

    '''
    add pixel to BFS init_list
    '''
    flag_matrix = np.zeros((height, width),)
    # 将边缘中这5种灰度值的点添加进BFS队列作为起始点灰度值按照接近黑色或白色：记flag为1和2
    init_list = []

    for x in range(width):
        for y in range(height):
            if x == 0 or y == 0 or x == 127 or y == 127:
                if img[x, y] in flag_array:
                    init_list.append([x, y])
                    if img[x, y] < 30:
                        flag_matrix[x, y] = 1
                    else:
                        flag_matrix[x, y] = 2
    '''
    BFS
    '''
    v = {(x, y): False for x in range(0, width) for y in range(0, height)}
    for init_pixel in init_list:
        flag_matrix = BFS(init_pixel[0], init_pixel[1], flag_matrix, img, v, global_v, thres)

    '''
    若 flag_matrix == 0, 统计该区块周围像素点的 flag，若flag同时含有1和2，或者是仅含有1：删除这些区块
    '''
    mask = np.zeros((height, width),)                
    mask[flag_matrix == 0] = 255

    # binary_img, num = skimage.measure.label(mask, neighbors=8, background=0, return_num=True)
    binary_img, num = skimage.measure.label(mask, connectivity=2, background=0, return_num=True)

    rows, cols = binary_img.shape
    # 对于每一个区块，检查周围flag
    cdef int row
    cdef int col
    for i in range(1, num + 1):
        neighbor_flag_list = []
        for row in range(rows):
            for col in range(cols):
                if binary_img[row, col] == i:
                    neighbor_flag_list = neighbor_flag(row, col, neighbor_flag_list, flag_matrix)
        if len(neighbor_flag_list) > 0:
            if min(neighbor_flag_list) == 1 or np.sum(binary_img == i) <= thr_num:
                flag_matrix[binary_img == i] = 3

    mask = np.zeros((height, width),)                
    mask[flag_matrix == 0] = 255  
    return mask