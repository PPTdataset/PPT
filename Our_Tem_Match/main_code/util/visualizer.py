import numpy as np
import os
import ntpath
import time
from . import util
from subprocess import Popen, PIPE
import sys
import torch.nn.functional as F
import json
from tools.contrast_image import direct_contrast
from tools.json_generator import json_generator, json_generator_all
from tools.utils import (Combine_img, Save, img2contours, contours2cont_max, is_inside_polygon, bgr2gray, gray2bgr, diffimage, Random, compare_defect)
import cv2

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


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
        # Pred_img = F.avg_pool2d(Pred_img, 2, stride=2, padding=0).squeeze() # [128, 128]

        point_list = (Pred_img > opt.sigmoid_thr).nonzero()  # [num, 2]
        height, width = Pred_img.shape

        # creat new dict
        new_dict = {'Height': {}, 'Width': {}, 'name': {}, 'regions': []}

        new_dict['Height'] = height
        new_dict['Width'] = width
        new_dict['name'] = "%s.bmp" % name

        new_points = {'points': []}
        for point in point_list:
            new_points['points'].append("%d, %d" % (point[0], point[1]))

        if len(new_points['points']) > 0:
            new_dict['regions'].append(new_points)
            new_json = json.dumps(new_dict)  # dict2json
            save_dir = opt.val_js_path if opt.is_val else opt.results_dir
            f = open(save_dir + '%s.json' % name, 'w')
            f.write(new_json)
            f.close()

        if opt.save_img:
            # 在这里添加比较算法
            TC_img = util.tensor2im(visuals['TC_img'], is_sigmoid=opt.is_sigmoid)
            Pred_img = util.tensor2im(visuals['Pred_img'], is_sigmoid=opt.is_sigmoid)
            Match_img = util.tensor2im(visuals['Match_img'], is_sigmoid=opt.is_sigmoid)
            Mask = util.tensor2im(visuals['Mask'], is_sigmoid=opt.is_sigmoid)

            # 调整图像大小至128
            # TC_img = cv2.resize(TC_img, (128, 128))
            # Pred_img = cv2.resize(Pred_img, (128, 128))
            # Match_img = cv2.resize(Match_img, (128, 128))

            if opt.input_nc == 1:
                TC_img = bgr2gray(TC_img)
                Pred_img = bgr2gray(Pred_img)
                Match_img = bgr2gray(Match_img)

            gt_path = os.path.join(opt.val_gt_path, "%s.bmp" % name)
            draw_imgs = [TC_img, Match_img, Mask, Pred_img, compare_defect(Pred_img, 127), compare_defect(Pred_img, 224)]

            if os.path.exists(gt_path):
                ground_truth = cv2.imread(gt_path)
                draw_imgs.append(ground_truth)

            conjunction = Combine_img(opt.input_nc, draw_imgs)
            Save(conjunction, name, "%s/temp_data/saves/%s" % (opt.ROOT, opt.part_name))

            if opt.print_log:
                print("save~")


def save_images_seg_v7(opt, visuals, image_path):
    # image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    # 在这里添加比较算法
    TC_img = util.tensor2im(visuals['TC_img'], is_sigmoid=opt.is_sigmoid)
    Pred_img = util.tensor2im(visuals['Pred_img'], is_sigmoid=opt.is_sigmoid)
    Match_img = util.tensor2im(visuals['Match_img'], is_sigmoid=opt.is_sigmoid)

    # 调整图像大小至128
    TC_img = cv2.resize(TC_img, (128, 128))
    Pred_img = cv2.resize(Pred_img, (128, 128))
    Match_img = cv2.resize(Match_img, (128, 128))

    if opt.input_nc == 1:
        TC_img = bgr2gray(TC_img)
        Pred_img = bgr2gray(Pred_img)
        Match_img = bgr2gray(Match_img)

    mask_thr = compare_defect(Pred_img, opt.contrast_thr)

    Mask = bgr2gray(np.array(visuals['Mask'].squeeze().cpu()))
    mask_thr[Mask == 0] = 0

    if opt.save_img:
        conjunction = Combine_img(opt.input_nc, [
            TC_img, Match_img, Mask, Pred_img,
            compare_defect(Pred_img, 127),
            compare_defect(Pred_img, 150),
            compare_defect(Pred_img, 180),
            compare_defect(Pred_img, 200),
            compare_defect(Pred_img, 210),
            compare_defect(Pred_img, 220), mask_thr
        ])
        Save(conjunction, "%s" % name, "%s/../saves/%s" % (opt.checkpoints_dir, opt.dataroot.split('/')[-3]))
        # cv2.imwrite('./saves/%s_fake.bmp' % name, image_fake)
        # cv2.imwrite('./saves/%s_real.bmp' % name, image_real)
        if opt.print_log:
            print("save~")

    if opt.save_json:
        # new_json = json_generator(mask_thr, name + ".bmp", thr_num=20)
        new_json = json_generator_all(mask_thr, name + ".bmp")

        if opt.is_val:
            if new_json is not None:
                f = open(opt.val_js_path + '%s.json' % name, 'w')
                f.write(new_json)
                f.close()
        else:
            if new_json is not None:
                f = open(opt.results_dir + '%s.json' % name, 'w')
                f.write(new_json)
                f.close()
    else:
        cv2.imwrite("./temp_data/segementation/%s.bmp" % name, mask_thr)


def save_images_seg_v5(opt, webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    # image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    # 在这里添加比较算法
    TC_img = util.tensor2im(visuals['TC_img'], is_sigmoid=opt.is_sigmoid)
    Pred_img = util.tensor2im(visuals['Pred_img'], is_sigmoid=opt.is_sigmoid)

    # 调整图像大小至128
    TC_img = cv2.resize(TC_img, (128, 128))
    Pred_img = cv2.resize(Pred_img, (128, 128))

    if opt.input_nc == 1:
        TC_img = bgr2gray(TC_img)
        Pred_img = bgr2gray(Pred_img)

    mask_thr = compare_defect(Pred_img, opt.contrast_thr)

    if opt.save_img:
        conjunction = Combine_img(opt.input_nc, [
            TC_img, Pred_img,
            compare_defect(Pred_img, 127),
            compare_defect(Pred_img, 150),
            compare_defect(Pred_img, 180),
            compare_defect(Pred_img, 200),
            compare_defect(Pred_img, 210),
            compare_defect(Pred_img, 220), mask_thr
        ])

        Save(conjunction, "%s" % name, "%s/../saves/%s" % (opt.checkpoints_dir, opt.dataroot.split('/')[-3]))
        # cv2.imwrite('./saves/%s_fake.bmp' % name, image_fake)
        # cv2.imwrite('./saves/%s_real.bmp' % name, image_real)
        if opt.print_log:
            print("save~")

    if opt.save_json:
        new_json = json_generator(mask_thr, name + ".bmp", thr_num=20)
        # new_json = json_generator_all(mask_thr, name + ".bmp")

        if opt.is_val:
            if new_json is not None:
                f = open(opt.val_js_path + '%s.json' % name, 'w')
                f.write(new_json)
                f.close()
        else:
            if new_json is not None:
                f = open(opt.results_dir + '%s.json' % name, 'w')
                f.write(new_json)
                f.close()
    else:
        cv2.imwrite("./temp_data/segementation/%s.bmp" % name, mask_thr)


def save_images_seg_v2(opt, webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    # image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    # 在这里添加比较算法
    ok = util.tensor2im(visuals[0], is_sigmoid=opt.is_sigmoid)
    pred = util.tensor2im(visuals[1], is_sigmoid=opt.is_sigmoid)
    df = util.tensor2im(visuals[2], is_sigmoid=opt.is_sigmoid)

    # 调整图像大小至128
    ok = cv2.resize(ok, (128, 128))
    pred = cv2.resize(pred, (128, 128))
    df = cv2.resize(df, (128, 128))

    if opt.input_nc == 1:
        ok = bgr2gray(ok)
        pred = bgr2gray(pred)
        df = bgr2gray(df)

    mask_thr = compare_defect(pred, opt.contrast_thr)

    if opt.save_img:
        conjunction = Combine_img(opt.input_nc, [
            df, ok, pred,
            compare_defect(pred, 127),
            compare_defect(pred, 150),
            compare_defect(pred, 180),
            compare_defect(pred, 200),
            compare_defect(pred, 210),
            compare_defect(pred, 220), mask_thr
        ])

        Save(conjunction, "%s" % name, "%s/../saves/%s" % (opt.checkpoints_dir, opt.dataroot.split('/')[-3]))
        # cv2.imwrite('./saves/%s_fake.bmp' % name, image_fake)
        # cv2.imwrite('./saves/%s_real.bmp' % name, image_real)
        if opt.print_log:
            print("save~")

    if opt.save_json:
        new_json = json_generator(mask_thr, name + ".bmp", thr_num=20)
        # new_json = json_generator_all(mask_thr, name + ".bmp")
        val_js_path = os.path.join(opt.val_pre_path, opt.name, "%s_%s" % (opt.defect_generator, opt.version),
                                   "%s_thr=%d/" % (opt.val_gt_path.split("/")[-2], opt.contrast_thr))
        results_dir = opt.results_dir

        if opt.is_val:
            if new_json is not None:
                f = open(val_js_path + '%s.json' % name, 'w')
                f.write(new_json)
                f.close()
        else:
            if new_json is not None:
                f = open(results_dir + '%s.json' % name, 'w')
                f.write(new_json)
                f.close()
    else:
        cv2.imwrite("./temp_data/segementation/%s.bmp" % name, mask_thr)


def save_images(opt, webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    # image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    # webpage.add_header(name)
    # ims, txts, links = [], [], []

    # save_path = os.path.join(image_dir, name + '.bmp')
    # for label, im_data in visuals.items():
    #     im = util.tensor2im(im_data)
    #     image_name = '%s_%s.png' % (name, label)
    #     save_path = os.path.join(image_dir, image_name)
    #     util.save_image(im, save_path, aspect_ratio=aspect_ratio)
    #     ims.append(image_name)
    #     txts.append(label)
    #     links.append(image_name)
    # webpage.add_images(ims, txts, links, width=width)

    # 在这里添加比较算法
    image_fake = util.tensor2im(visuals['fake'], is_sigmoid=opt.is_sigmoid)
    image_real = util.tensor2im(visuals['real'], is_sigmoid=opt.is_sigmoid)

    # 调整图像大小至128
    image_real = cv2.resize(image_real, (128, 128))
    image_fake = cv2.resize(image_fake, (128, 128))
    if opt.input_nc == 1:
        image_real = bgr2gray(image_real)
        image_fake = bgr2gray(image_fake)

    contrast_image = globals()["direct_contrast"]
    # contrast_image = globals()["LBP"]
    mask_thr, mask_pro = contrast_image(image_real, image_fake, thr=opt.contrast_thr)

    width, height = image_real.shape[:2]

    if opt.use_mask and "part2" in opt.name:  # for part 2
        # image_fake 检测下，把黑色部分去掉
        test_img = np.zeros((width, height), image_real.dtype)
        dst, contours = img2contours(image_real, thr=20)
        if len(contours) > 0:
            for i in range(width):
                for j in range(height):
                    for cont in contours:
                        flag = cv2.pointPolygonTest(cont, (i, j), False)
                        if flag == 1:
                            test_img[j, i] = 1
            mask_thr[test_img == 0] = 0
            # 画边界
            # contour_img = gray2bgr(image_real).copy()
            # contour_img = cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 1)
            # cv2.imwrite("./saves/%s_contour.bmp"% name, contour_img)
            # cv2.imwrite("./saves/%s_test_img.bmp"% name, test_img)

    elif opt.use_mask and "part1" in opt.name:  # for part 1
        # 二值化img_real，找出黑色部分，并删除mask中对应的部分。
        test_img = np.zeros((width, height), image_real.dtype)
        dst, contours = img2contours(image_real, thr=70)  # 40
        if len(contours) > 0:
            cont = contours2cont_max(contours)
            for i in range(width):
                for j in range(height):
                    flag = cv2.pointPolygonTest(cont, (i, j), False)
                    if flag == 1:
                        test_img[j, i] = 1
            mask_thr[test_img == 0] = 0  # 保留白色

            # 画边界
            # contour_img = gray2bgr(image_real).copy()
            # contour_img = cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 1)
            # cv2.imwrite("./saves/%s_contour.bmp"% name, contour_img)
            # cv2.imwrite("./saves/%s_test_img.bmp"% name, test_img)

    elif opt.use_mask and "part2" in opt.name and ("white" in opt.name or "black" in opt.name):  # for part 2 two mask
        # 二值化img_real，分别针对保留黑色部分和保留白色部分
        test_img = np.zeros((width, height), image_real.dtype)
        dst, contours = img2contours(image_real, thr=200)  # thr = 50
        if len(contours) > 0:
            cont = contours2cont_max(contours)
            for i in range(width):
                for j in range(height):
                    flag = cv2.pointPolygonTest(cont, (i, j), False)
                    if flag == 1:
                        test_img[j, i] = 1
            if "white" in opt.name:
                mask_thr[test_img == 0] = 0  # 保留白色
            elif "black" in opt.name:
                mask_thr[test_img == 1] = 0  # 保留黑色

            # 画边界
            # contour_img = gray2bgr(image_real).copy()
            # contour_img = cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 1)
            # cv2.imwrite("./saves/%s_contour.bmp"% name, contour_img)
            # cv2.imwrite("./saves/%s_test_img.bmp"% name, test_img)

    if opt.save_img:
        # if opt.save_img and name == "0aAQrpCu416rlL9HLs6uH2696vIMyk":
        conjunction = Combine_img(opt.input_nc, [
            image_real, image_fake, mask_pro,
            contrast_image(image_real, image_fake, thr=20)[0],
            contrast_image(image_real, image_fake, thr=25)[0],
            contrast_image(image_real, image_fake, thr=30)[0],
            contrast_image(image_real, image_fake, thr=35)[0],
            contrast_image(image_real, image_fake, thr=40)[0],
            contrast_image(image_real, image_fake, thr=45)[0], mask_thr
        ])

        Save(conjunction, "%s" % name, "%s/../saves/" % opt.checkpoints_dir)
        # cv2.imwrite('./saves/%s_fake.bmp' % name, image_fake)
        # cv2.imwrite('./saves/%s_real.bmp' % name, image_real)
        if opt.print_log:
            print("save~")

    if opt.save_json:
        # new_json = json_generator(mask_thr, name + ".bmp", thr_num=0)
        new_json = json_generator_all(mask_thr, name + ".bmp")
        val_js_path = os.path.join(opt.val_pre_path, opt.name, "%s_%s" % (opt.defect_generator, opt.version),
                                   "%s_thr=%d/" % (opt.val_gt_path.split("/")[-2], opt.contrast_thr))
        results_dir = opt.results_dir

        if opt.is_val:
            if new_json is not None:
                f = open(val_js_path + '%s.json' % name, 'w')
                f.write(new_json)
                f.close()
        else:
            if new_json is not None:
                f = open(results_dir + '%s.json' % name, 'w')
                f.write(new_json)
                f.close()
    else:
        cv2.imwrite("./temp_data/segementation/%s.bmp" % name, mask_thr)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """
    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options 
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir + opt.name + "_%s/" % (opt.defect_generator), 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:  # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1, padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2, opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:  # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label), win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            from . import html
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                          Y=np.array(self.plot_data['Y']),
                          opts={
                              'title': self.name + ' loss over time',
                              'legend': self.plot_data['legend'],
                              'xlabel': 'epoch',
                              'ylabel': 'loss'
                          },
                          win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, dataset_size, losses, t_comp, t_data, lr=0, miou=0., eta=0.):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        day = eta // (24 * 60 * 60)
        hour = (eta % (24 * 60 * 60)) // (60 * 60)
        minute = (eta % (60 * 60)) // 60
        second = eta % 60
        message = '(epoch: %d, iters: %d/%d, time: %.7f, data: %.7f, lr: %.7f, miou: %.7f, eta: %dd %dh %dm %ds) ' % (epoch, iters, dataset_size, t_comp, t_data, lr,
                                                                                                                      miou, day, hour, minute, second)
        for k, v in losses.items():
            message += '%s: %.10f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
