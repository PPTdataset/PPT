# -*- coding: utf-8 -*-
ROOT = '..'

import sys
#sys.path.append('%s/main_code' % ROOT)
import torch
print(torch.__version__)
import os
import ntpath
import time
from PIL import Image
from options.test_options import TestOptions
from train import train
from options.main_options import *
from data import create_dataset
from models import create_model
from tools.save_json import save_images_seg_v8,save_images_seg_v9
from tools.validate import validate, validate_v2, validate_check
from tools.utils import Make_dirs
from zip_file import zipdir
from util import util
from data.base_dataset import BaseDataset, get_transform


class data_prefetcher():
    def __init__(self, loader, opt):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc

        self.mean = torch.tensor([255.]).cuda()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return

        with torch.cuda.stream(self.stream):
            for key in self.next_data:
                if key != 'TC_path':
                    self.next_data[key] = self.next_data[key].cuda(non_blocking=True)
                    # self.next_data[key] = self.next_data[key] / self.mean
                    # self.next_data[key] = self.next_data[key].permute(0, 3, 1, 2)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        if data is not None:
            for key in data:
                if key != 'TC_path':
                    data[key].record_stream(torch.cuda.current_stream())

        self.preload()
        return data


def test(TC_property):
    
    init_time = time.time()
    opt = TestOptions().parse()
    opt.ROOT = ROOT
    opt = base_parse(opt)
    opt = test_parse_base(opt)

    print("************** start test %s **************" % opt.model_name)

    opt.TC_property = TC_property

    print("sigmoid_thr = ", opt.sigmoid_thr)
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name+"_%s" % (opt.defect_generator))
    opt.expr_dir = expr_dir
    util.mkdirs(expr_dir)

    save_images = globals()[opt.save_images_versoin]
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

    print('------------------------ Network initialized ------------------------')
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    print('------------------------ Dataset initialized ------------------------')
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print('---------------------------------------------------------------------')

    opt.val_js_path = os.path.join(opt.val_pre_path, opt.name, "sigmoid_thr=%f/" % opt.sigmoid_thr)
    if opt.is_val:
        if os.path.exists(opt.val_js_path):
            import shutil
            shutil.rmtree(opt.val_js_path)
        Make_dirs(opt.val_js_path)
    else:
        Make_dirs(opt.results_dir)

    if opt.eval:
        model.eval()

    prefetcher = data_prefetcher(dataset, opt)
    data = prefetcher.next()
    print("init_time =", time.time() - init_time)
    inference_time = time.time()
    i = 0
    while data is not None:
        # only apply our model to opt.num_test images.
        if i >= opt.num_test:
            break
        model.set_input(data)

        img_path = model.get_image_paths()

        if i % opt.batch_size == 0 and opt.print_log:
            print('processing (%04d)-th image... %s' % (i, img_path[0]))

        model.test()
        visuals = model.get_current_visuals()
        save_images(opt, visuals, img_path)

        data = prefetcher.next()
        i += opt.batch_size

    inference_time = time.time() - inference_time
    if opt.is_val:#验证并计算最后得分
        validate_v2(opt.val_js_path, opt.val_gt_path, ROOT, opt)
        # validate_check(opt.val_js_path, opt.val_gt_path, ROOT, opt)
    # 训练结果opt.val_js_path = ..\temp_data\results_validation\part3_v10_24e1t10r4_eps1_plus40\sigmoid_thr=0.500000
    # 人工标注opt.val_gt_path = '%s/val_data/part3/' % ROOT
    print("inference_time =", inference_time)
    print(opt.name)


if __name__ == '__main__':
    all_start_time = time.time()
    '''SAVEIMG = "%s/temp_data/saves" % ROOT
    RESULTS = "%s/temp_data/result" % ROOT
    if os.path.exists(SAVEIMG):
        import shutil
        shutil.rmtree(SAVEIMG)
    Make_dirs(SAVEIMG)
    if os.path.exists(RESULTS):
        import shutil
        shutil.rmtree(RESULTS)'''
    print("start")
    # train
    TC_property = None
    TC_property = train(all_start_time)
    # test
    test(TC_property)

    #zipdir("%s/temp_data/result/" % ROOT, "%s/result/data.zip" % ROOT)

    total_time = time.time() - all_start_time
    day = total_time // (24 * 60 * 60)
    hour = (total_time % (24 * 60 * 60)) // (60 * 60)
    minute = (total_time % (60 * 60)) // 60
    second = total_time % 60
    print("total_time: %dd %dh %dm %ds" % (day, hour, minute, second))
