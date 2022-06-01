# -*- coding: utf-8 -*-
ROOT = '..'

import sys
import time
import os
import torch
from options.train_options import TrainOptions
from options.main_options import *
from data import create_dataset
from models import create_model  #code/models
from util.visualizer import Visualizer
from util import util

EPS = 1e-5


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
                self.next_data[key] = self.next_data[key].cuda(non_blocking=True)
                # self.next_data[key] = self.next_data[key] / self.mean
                # self.next_data[key] = self.next_data[key].permute(0, 3, 1, 2)
                # print(self.next_data[key].shape)
                # print(self.next_data[key])

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        if data is not None:
            for key in data:
                data[key].record_stream(torch.cuda.current_stream())

        self.preload()
        return data


def train(all_start_time):
    # get training options
    
    # init_time = time.time()
    opt = TrainOptions().parse()
    opt = base_parse(opt)
    opt.mother_img = None

    opt = train_parse_base(opt)

    print("************** start train %s **************" % opt.model_name)

    opt.ROOT = ROOT

    opt = base_parse(opt)

    expr_dir = os.path.join(opt.checkpoints_dir, opt.name+"_%s" % (opt.defect_generator))
    opt.expr_dir = expr_dir
    util.mkdirs(expr_dir)
    save_check_dir = '%s/temp_data/debug/%s' % (ROOT, opt.name)
    util.mkdirs(save_check_dir)
    print(opt.name)
    print('------------------------ Network initialized ------------------------')
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    #opt.model = 'seg_v10'  opt.netG = 'seg_v10'
    #/data/seg_v10_plus_dataset
    print('------------------------ Dataset initialized ------------------------')
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    all_data_size = dataset_size * opt.n_epochs
    print('The number of training images = %d' % dataset_size)
    print('---------------------------------------------------------------------')

    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    single_image_time = None
    print_num = 0

    # print("init_time =", time.time() - init_time)
    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        iter_data_time = time.time()  # timer for data loading per iteration
        print_start_time = time.time()  # timer for computation per iteration

        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        prefetcher = data_prefetcher(dataset, opt)  #预加载用的？
        data = prefetcher.next()
        
        # print("epoch_init =", time.time() - iter_data_time)
        epoch_start_time = time.time()  # timer for entire epoch
        while data is not None:
            if total_iters % opt.print_freq == 0:
                t_data = time.time() - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0 and opt.print_log:  # print training losses and save logging information to the disk
                miou = model.get_miou()
                losses = model.get_current_losses()
                t_comp = time.time() - print_start_time
                print_start_time = time.time()  # timer for computation per iteration
                single_image_time = t_comp / opt.print_freq if single_image_time is None else (single_image_time * print_num + t_comp / opt.print_freq) / (print_num + 1)
                print_num += 1
                eta_images = all_data_size - total_iters
                eta = single_image_time * eta_images
                visualizer.print_current_losses(epoch, epoch_iter, dataset_size, losses, t_comp, t_data, model.optimizers[0].param_groups[0]['lr'], miou, eta)
                print("not_gen_defect_rate =", dataset.dataset.not_gen_defect / (dataset.dataset.not_gen_defect + dataset.dataset.gen_defect))
                print("single_image_time =", single_image_time)
                print("FPS =", 1. / single_image_time)
                print('---------------------------------------------------------------------')
                # if 'check' in opt.dataset_mode and print_num % 3 == 0:
                #     break

            if total_iters % opt.check_img_freq == 0 and opt.save_check_img:
                model.save_check("epoch_img_%s" % total_iters, save_check_dir)

            # if time is not enough
            if (time.time() - all_start_time) > opt.limit_time:
                print("************* time up for %s s*************" % opt.limit_time)
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                break

            iter_data_time = time.time()
            data = prefetcher.next()

        one_epoch_time = 'End of epoch %d / %d \t Time Taken: %.7f sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time)
        print(one_epoch_time)

        save_and_log_time = time.time()
        # cache our model every <save_epoch_freq> epochs
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            # model.save_networks(epoch)

        log_path = os.path.join(opt.checkpoints_dir + opt.name + "_%s/" % (opt.defect_generator), 'loss_log.txt')
        f = open(log_path, "a")
        f.write(one_epoch_time + '\n')
        f.close()

        # update learning rates at the end of every epoch.
        model.update_learning_rate()
        print("save_and_log_time =", time.time() - save_and_log_time)

    print(opt.name[6:])
    return dataset.dataset.TC_property