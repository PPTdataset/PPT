import cv2
import numpy as np
import random
from .utils import Save, diffimage


def Clip(x, min_x, max_x):
    return max(min(x, max_x), min_x)


def seamless_clone_defect_generator(opt, OK_img, TC_img, use_same_tc, keypoints_tc_match):
    assert OK_img.shape == TC_img.shape
    show_mask = np.zeros(OK_img.shape, OK_img.dtype)
    DF_img_big = np.zeros(OK_img.shape, OK_img.dtype)

    # 得到去黑边的图
    l, t, r, b = keypoints_tc_match
    if min(b - t, r - l) < 30:
        # TODO: 如果有效区域太少......
        return OK_img.copy(), np.zeros(OK_img.shape, OK_img.dtype)
    OK_img = OK_img[t:b, l:r]
    if use_same_tc:
        TC_img = TC_img[t:b, l:r]
        assert OK_img.shape == TC_img.shape
    height, width = OK_img.shape[:2]

    # 自己设计
    if not use_same_tc and random.random() < opt.nootc:
        # 多边形
        if random.random() < 0.5:
            # if 0:
            defect_h = random.randint(10, height // 3)
            defect_w = random.randint(10, width // 3)
            defect = np.zeros((defect_h, defect_w, 3), OK_img.dtype)
            polygon = np.array([[random.randrange(0, defect_h // 2), random.randrange(0, defect_w // 2)],
                                [random.randrange(0, defect_h // 2), random.randrange(defect_w // 2, defect_w)],
                                [random.randrange(defect_h // 2, defect_h), random.randrange(defect_w // 2, defect_w)],
                                [random.randrange(defect_h // 2, defect_h), random.randrange(0, defect_w // 2)]])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            defect = cv2.fillConvexPoly(defect, polygon, color)

            # 模糊瑕疵
            blur_kernel = random.randrange(7, 11, 2)
            defect = cv2.GaussianBlur(defect, (blur_kernel, blur_kernel), 0)
        # 折线段（可能只有一条线）
        else:
            defect_h = height
            defect_w = width
            defect = np.zeros((defect_h, defect_w, 3), OK_img.dtype)
            point_list = [(random.randrange(0, defect_h), random.randrange(0, defect_w)), (random.randrange(0, defect_h), random.randrange(0, defect_w)),
                          (random.randrange(0, defect_h), random.randrange(0, defect_w))]
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            defect = cv2.line(defect, point_list[0], point_list[1], color, random.randrange(1, 4))
            if random.randint(0, 1):
                defect = cv2.line(defect, point_list[1], point_list[2], color, random.randrange(1, 4))
            # 模糊瑕疵
            blur_kernel = random.randrange(3, 7, 2)
            defect = cv2.GaussianBlur(defect, (blur_kernel, blur_kernel), 0)

        # 瑕疵仿射变换，旋转等
        mat_rotate = cv2.getRotationMatrix2D(center=(defect.shape[0] * 0.5, defect.shape[1] * 0.5), angle=random.randrange(0, 360), scale=1)
        defect = cv2.warpAffine(defect, mat_rotate, (defect.shape[1], defect.shape[0]))
        mask = defect.copy()

        top = random.randint(0, height - defect_h)
        left = random.randint(0, width - defect_w)
        OK_img_s = OK_img[top:top + defect_h, left:left + defect_w].astype(np.int)
        defect = defect.astype(np.int)

        # 相加或相减，再截断到0-255之间，再赋值回原区域
        OK_img_s = OK_img_s - defect if random.random() > 0.5 else OK_img_s + defect
        OK_img_s = np.clip(OK_img_s, 0, 255).astype(np.uint8)
        OK_img[top:top + defect_h, left:left + defect_w] = OK_img_s
        DF_img_big[t:b, l:r] = OK_img

        show_mask[t + top:t + top + defect_h, l + left:l + left + defect_w] = mask
        return DF_img_big, show_mask

    area = height * width
    if use_same_tc:
        min_crop = min(height, width) // 2
        max_area = area // 2
    else:
        min_crop = 20
        max_area = area // 4

    defect_area = random.randint(min_crop**2, max_area)

    defect_height = random.randint(min_crop, min(defect_area // min_crop, height))
    defect_width = min(defect_area // defect_height, width)

    top = random.randint(0, height - defect_height)
    left = random.randint(0, width - defect_width)

    defect = TC_img[top:top + defect_height, left:left + defect_width]

    if use_same_tc:
        OK_img_s = OK_img[top:top + defect_height, left:left + defect_width].copy().astype(np.int)

        if random.random() < 0.5:
            color = OK_img[height // 2, width // 2]
        else:
            color = OK_img_s[random.randint(0, defect_height - 1), random.randint(0, defect_width - 1)]

        OK_img_s = np.abs(OK_img_s - color)
        OK_img_s = np.mean(OK_img_s, axis=2)
        mask = np.zeros(defect.shape, defect.dtype)
        low = OK_img_s < opt.region_thr
        mask[low] = 255

        # 保底策略
        mask_h = random.randint(10, min(20, defect_height))
        mask_w = random.randint(10, min(20, defect_width))
        mask_top = random.randint(0, defect_height - mask_h)
        mask_left = random.randint(0, defect_width - mask_w)
        mask[mask_top:mask_top + mask_h, mask_left:mask_left + mask_w] = 255
    else:
        # Create an all white mask
        mask = 255 * np.ones(defect.shape, defect.dtype)

    # TODO：找到自己就是对齐的贴
    if use_same_tc:
        paste_center_h = top + (defect_height) // 2 - min(7, top)
        paste_center_w = left + (defect_width) // 2
    else:
        paste_center_h = random.randint(defect_height // 2, height - defect_height // 2)
        paste_center_w = random.randint(defect_width // 2, width - defect_width // 2)

    center = (paste_center_w, paste_center_h)

    try:
        # TODO NORMAL_CLONE or MIXED_CLONE
        # DF_img = cv2.seamlessClone(defect, OK_img, mask, center, cv2.NORMAL_CLONE)
        DF_img = cv2.seamlessClone(defect, OK_img, mask, center, cv2.MIXED_CLONE)
    except:
        Save(mask, '_mask')
        print(defect.shape)
        print(OK_img.shape)
        print(center)
        print(use_same_tc)

    if use_same_tc:
        show_mask[t + top:t + top + defect_height, l + left:l + left + defect_width] = mask
    else:
        show_mask[top:top + defect_height, left:left + defect_width] = mask

    # 把DF放回大图
    DF_img_big[t:b, l:r] = DF_img
    return DF_img_big, show_mask


def seamless_clone_defect_generator_plus(opt, OK_img, TC_img, use_same_tc, keypoints_tc_match):
    assert OK_img.shape == TC_img.shape
    show_mask = np.zeros(OK_img.shape, OK_img.dtype)
    DF_img_big = np.zeros(OK_img.shape, OK_img.dtype)

    # 得到去黑边的图
    l, t, r, b = keypoints_tc_match
    if min(b - t, r - l) < 30:
        # TODO: 如果有效区域太少......
        return OK_img.copy(), np.zeros(OK_img.shape, OK_img.dtype)
    OK_img = OK_img[t:b, l:r]
    if use_same_tc:
        TC_img = TC_img[t:b, l:r]
        assert OK_img.shape == TC_img.shape
    height, width = OK_img.shape[:2]

    area = height * width
    if use_same_tc:
        min_crop = min(height, width) // 2
        max_area = area // 2
    else:
        min_crop = 20
        max_area = area // 4

    defect_area = random.randint(min_crop**2, max_area)

    defect_height = random.randint(min_crop, min(defect_area // min_crop, height))
    defect_width = min(defect_area // defect_height, width)

    top = random.randint(0, height - defect_height)
    left = random.randint(0, width - defect_width)

    defect = TC_img[top:top + defect_height, left:left + defect_width]

    if use_same_tc:
        OK_img_s = OK_img[top:top + defect_height, left:left + defect_width].copy().astype(np.int)

        if random.random() < 0.5:
            color = OK_img[height // 2, width // 2]
        else:
            color = OK_img_s[random.randint(0, defect_height - 1), random.randint(0, defect_width - 1)]

        OK_img_s = np.abs(OK_img_s - color)
        OK_img_s = np.mean(OK_img_s, axis=2)
        mask = np.zeros(defect.shape, defect.dtype)
        low = OK_img_s < opt.region_thr
        mask[low] = 255

        # 保底策略
        mask_h = random.randint(10, min(20, defect_height))
        mask_w = random.randint(10, min(20, defect_width))
        mask_top = random.randint(0, defect_height - mask_h)
        mask_left = random.randint(0, defect_width - mask_w)
        mask[mask_top:mask_top + mask_h, mask_left:mask_left + mask_w] = 255
    else:
        # Create an all white mask
        mask = 255 * np.ones(defect.shape, defect.dtype)

    # TODO：找到自己就是对齐的贴
    if use_same_tc:
        paste_center_h = top + (defect_height) // 2 - min(7, top)
        paste_center_w = left + (defect_width) // 2
    else:
        paste_center_h = random.randint(defect_height // 2, height - defect_height // 2)
        paste_center_w = random.randint(defect_width // 2, width - defect_width // 2)

    center = (paste_center_w, paste_center_h)

    try:
        # TODO NORMAL_CLONE or MIXED_CLONE
        # DF_img = cv2.seamlessClone(defect, OK_img, mask, center, cv2.NORMAL_CLONE)
        DF_img = cv2.seamlessClone(defect, OK_img, mask, center, cv2.MIXED_CLONE)
    except:
        Save(mask, '_mask')
        print(defect.shape)
        print(OK_img.shape)
        print(center)
        print(use_same_tc)

    if use_same_tc:
        show_mask[t + top:t + top + defect_height, l + left:l + left + defect_width] = mask
    else:
        show_mask[top:top + defect_height, left:left + defect_width] = mask

    Label = diffimage(OK_img, DF_img, opt.input_nc == 1)
    Label[Label > opt.contrast_thr] = 255
    Label[Label <= opt.contrast_thr] = 0
    if Label.sum() > 0:
        # 把DF放回大图
        DF_img_big[t:b, l:r] = DF_img
        return DF_img_big, show_mask
    else:
        # 多边形
        if random.random() < 0.5:
            defect_h = random.randint(10, height // 3)
            defect_w = random.randint(10, width // 3)
            defect = np.zeros((defect_h, defect_w, 3), OK_img.dtype)
            polygon = np.array([[random.randrange(0, defect_h // 2), random.randrange(0, defect_w // 2)],
                                [random.randrange(0, defect_h // 2), random.randrange(defect_w // 2, defect_w)],
                                [random.randrange(defect_h // 2, defect_h), random.randrange(defect_w // 2, defect_w)],
                                [random.randrange(defect_h // 2, defect_h), random.randrange(0, defect_w // 2)]])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            defect = cv2.fillConvexPoly(defect, polygon, color)

            # 模糊瑕疵
            blur_kernel = random.randrange(7, 11, 2)
            defect = cv2.GaussianBlur(defect, (blur_kernel, blur_kernel), 0)
        # 折线段（可能只有一条线）
        else:
            defect_h = height
            defect_w = width
            defect = np.zeros((defect_h, defect_w, 3), OK_img.dtype)
            point_list = [(random.randrange(0, defect_h), random.randrange(0, defect_w)), (random.randrange(0, defect_h), random.randrange(0, defect_w)),
                          (random.randrange(0, defect_h), random.randrange(0, defect_w))]
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            defect = cv2.line(defect, point_list[0], point_list[1], color, random.randrange(1, 4))
            if random.randint(0, 1):
                defect = cv2.line(defect, point_list[1], point_list[2], color, random.randrange(1, 4))
            # 模糊瑕疵
            blur_kernel = random.randrange(3, 7, 2)
            defect = cv2.GaussianBlur(defect, (blur_kernel, blur_kernel), 0)

        # 瑕疵仿射变换，旋转等
        mat_rotate = cv2.getRotationMatrix2D(center=(defect.shape[0] * 0.5, defect.shape[1] * 0.5), angle=random.randrange(0, 360), scale=1)
        defect = cv2.warpAffine(defect, mat_rotate, (defect.shape[1], defect.shape[0]))
        mask = defect.copy()

        top = random.randint(0, height - defect_h)
        left = random.randint(0, width - defect_w)
        OK_img_s = OK_img[top:top + defect_h, left:left + defect_w].astype(np.int)
        defect = defect.astype(np.int)

        # 相加或相减，再截断到0-255之间，再赋值回原区域
        OK_img_s = OK_img_s - defect if random.random() > 0.5 else OK_img_s + defect
        OK_img_s = np.clip(OK_img_s, 0, 255).astype(np.uint8)
        OK_img[top:top + defect_h, left:left + defect_w] = OK_img_s
        DF_img_big[t:b, l:r] = OK_img

        show_mask[t + top:t + top + defect_h, l + left:l + left + defect_w] = mask
        return DF_img_big, show_mask


def seamless_clone_defect_generator_plus_ok(opt, OK_img, TC_img, use_same_tc, keypoints_tc_match):
    assert OK_img.shape == TC_img.shape
    show_mask = np.zeros(OK_img.shape, OK_img.dtype)
    DF_img_big = np.zeros(OK_img.shape, OK_img.dtype)

    # 得到去黑边的图
    l, t, r, b = keypoints_tc_match
    if min(b - t, r - l) < 30:
        # TODO: 如果有效区域太少......
        return OK_img.copy(), np.zeros(OK_img.shape, OK_img.dtype)
    OK_img = OK_img[t:b, l:r]
    height, width = OK_img.shape[:2]

    if random.random() < 0.6 and not opt.simple_defect:
        area = height * width
        if use_same_tc:
            min_crop = min(height, width) // 2
            max_area = area // 2
        else:
            min_crop = 20
            max_area = area // 4

        defect_area = random.randint(min_crop**2, max_area)

        defect_height = random.randint(min_crop, min(defect_area // min_crop, height))
        defect_width = min(defect_area // defect_height, width)

        top = random.randint(0, height - defect_height)
        left = random.randint(0, width - defect_width)

        defect = TC_img[top:top + defect_height, left:left + defect_width]

        if use_same_tc:
            OK_img_s = OK_img[top:top + defect_height, left:left + defect_width].copy().astype(np.int)

            if random.random() < 0.5:
                color = OK_img[height // 2, width // 2]
            else:
                color = OK_img_s[random.randint(0, defect_height - 1), random.randint(0, defect_width - 1)]

            OK_img_s = np.abs(OK_img_s - color)
            OK_img_s = np.mean(OK_img_s, axis=2)
            mask = np.zeros(defect.shape, defect.dtype)
            low = OK_img_s < opt.region_thr
            mask[low] = 255

            # 保底策略
            mask_h = random.randint(10, min(20, defect_height))
            mask_w = random.randint(10, min(20, defect_width))
            mask_top = random.randint(0, defect_height - mask_h)
            mask_left = random.randint(0, defect_width - mask_w)
            mask[mask_top:mask_top + mask_h, mask_left:mask_left + mask_w] = 255
        else:
            # Create an all white mask
            mask = 255 * np.ones(defect.shape, defect.dtype)

        # TODO：找到自己就是对齐的贴
        # if use_same_tc:
        #     paste_center_h = top + (defect_height) // 2 - min(7, top)
        #     paste_center_w = left + (defect_width) // 2
        # else:
        paste_center_h = random.randint(defect_height // 2, height - defect_height // 2)
        paste_center_w = random.randint(defect_width // 2, width - defect_width // 2)

        center = (paste_center_w, paste_center_h)

        try:
            # TODO NORMAL_CLONE or MIXED_CLONE
            # DF_img = cv2.seamlessClone(defect, OK_img, mask, center, cv2.NORMAL_CLONE)
            DF_img = cv2.seamlessClone(defect, OK_img, mask, center, cv2.MIXED_CLONE)
        except:
            Save(mask, '_mask')
            print(defect.shape)
            print(OK_img.shape)
            print(center)
            print(use_same_tc)
        # if mask.sum() == 0:
        #     print("2")
        show_mask[top:top + defect_height, left:left + defect_width] = mask

        Label = diffimage(OK_img, DF_img, opt.input_nc == 1)
        Label[Label > opt.contrast_thr] = 255
        Label[Label <= opt.contrast_thr] = 0
        if Label.sum() > 0:
            # 把DF放回大图
            DF_img_big[t:b, l:r] = DF_img
            return DF_img_big, show_mask

    # 多边形
    if random.random() < 0.5:
        # if 0:
        defect_h = random.randint(10, height // 3)
        defect_w = random.randint(10, width // 3)
        # defect_h = random.randrange(0, height)
        # defect_w = random.randrange(0, width)
        defect = np.zeros((defect_h, defect_w, 3), OK_img.dtype)
        # polypoint1 = [random.randrange(0, defect_h // 2), random.randrange(0, defect_w // 2)]

        polygon = np.array([[random.randrange(0, defect_h // 2), random.randrange(0, defect_w // 2)],
                            [random.randrange(0, defect_h // 2), random.randrange(defect_w // 2, defect_w)],
                            [random.randrange(defect_h // 2, defect_h), random.randrange(defect_w // 2, defect_w)],
                            [random.randrange(defect_h // 2, defect_h), random.randrange(0, defect_w // 2)]])
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        defect = cv2.fillConvexPoly(defect, polygon, color)
        # kkkkkkk1 = defect.copy()
        # 模糊瑕疵
        blur_kernel = random.randrange(7, 11, 2)
        defect = cv2.GaussianBlur(defect, (blur_kernel, blur_kernel), 0)
        # kkkkkkk2 = defect.copy()
    # 折线段（可能只有一条线）
    else:
        defect_h = height
        defect_w = width
        defect = np.zeros((defect_h, defect_w, 3), OK_img.dtype)
        point_list = [(random.randrange(0, defect_h), random.randrange(0, defect_w)), (random.randrange(0, defect_h), random.randrange(0, defect_w)),
                      (random.randrange(0, defect_h), random.randrange(0, defect_w))]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        defect = cv2.line(defect, point_list[0], point_list[1], color, random.randrange(1, 4))
        if random.randint(0, 1):
            defect = cv2.line(defect, point_list[1], point_list[2], color, random.randrange(1, 4))
        # 模糊瑕疵
        blur_kernel = random.randrange(3, 7, 2)
        defect = cv2.GaussianBlur(defect, (blur_kernel, blur_kernel), 0)

    # 瑕疵仿射变换，旋转等
    # mat_rotate = cv2.getRotationMatrix2D(center=(defect.shape[0] * 0.5, defect.shape[1] * 0.5), angle=random.randrange(0, 360), scale=1)
    # defect = cv2.warpAffine(defect, mat_rotate, (defect.shape[1], defect.shape[0]))

    mask = defect.copy()

    top = random.randint(0, height - defect_h)
    left = random.randint(0, width - defect_w)
    OK_img_s = OK_img[top:top + defect_h, left:left + defect_w].astype(np.int)
    defect = defect.astype(np.int)

    # 相加或相减，再截断到0-255之间，再赋值回原区域
    OK_img_s = OK_img_s - defect if random.random() > 0.5 else OK_img_s + defect
    OK_img_s = np.clip(OK_img_s, 0, 255).astype(np.uint8)
    OK_img[top:top + defect_h, left:left + defect_w] = OK_img_s
    DF_img_big[t:b, l:r] = OK_img

    # if mask.sum() == 0:
    #     print("1")
    #     print(polygon)
    #     r = random.randint(0, 999999999999999)
    #     Save(kkkkkkk1, r)
    #     Save(kkkkkkk2, r)

    show_mask[t + top:t + top + defect_h, l + left:l + left + defect_w] = mask
    return DF_img_big, show_mask
