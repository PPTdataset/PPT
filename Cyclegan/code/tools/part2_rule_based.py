import cv2
import numpy as np 

'''
******************** v1 ********************
'''
def part2_rule_based_v1(img):
    height, width = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化，取阈值为
    ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # 寻找图像中的轮廓
    contour_output = cv2.findContours(thresh, 2, 1)
    contours, hierarchy = contour_output[-2:]
    if len(contours) == 0: #若没有轮廓，跳过
        return np.zeros((height, width),)

    temp_image = img.copy()
    # drawContours(temp_image, contours, -1,  (0, 0, 255), 2)
    # cv2.imwrite('part2_contour.bmp', temp_image)
    # cv2.imshow('contour',temp_image)
    cont = contours[0]
    for x in range(temp_image.shape[0]):
        for y in range(temp_image.shape[1]):
            flag = cv2.pointPolygonTest(cont,(x,y),False)
            if flag == 1:
                temp_image[y,x] = 255
    contour_white = temp_image.copy()

    # 寻找物体的凸包并绘制凸包的轮廓
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        length = len(hull)

    convexhull_white = img.copy()
    for x in range(convexhull_white.shape[0]):
        for y in range(convexhull_white.shape[1]):
            flag = cv2.pointPolygonTest(hull, (x, y), False)
            if flag == 1:
                convexhull_white[y,x] = 255

    is_defect = (contour_white != convexhull_white)
    mask = np.zeros((height, width),)
    mask[is_defect] = 255

    return mask