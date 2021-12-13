# coding=utf-8

from model_fine_tune_discriminator import Fine_Tune_Disc

from opts_fine_tune_discriminator import parse_opts_ft
from opts import parse_opts
from dataloader import load_data_train
from dataloader import load_data
from sklearn import metrics
from model import OGNet
import numpy as np

def fine_tune():  #Phase two

    opt = parse_opts_ft()
    low_epoch = opt.low_epoch
    high_epoch = opt.high_epoch
    for i in range(high_epoch, high_epoch+1, 1):

        dataloader = load_data_train(opt)

        model = Fine_Tune_Disc(opt, dataloader)
        model.cuda()

        model_folder_path = './models/'
        load_model_epoch = ['epoch_{0}'.format(low_epoch), 'epoch_{0}'.format(i)]
        model.train(load_model_epoch, model_folder_path)

if __name__=="__main__":
    opt = parse_opts_ft()
    low_epoch = opt.low_epoch
    high_epoch = opt.high_epoch
    #for i in range(high_epoch, high_epoch+1, 1):
    print("stage 1 result:")
    high_epoch_g_model_name = 'g_high_epoch'
    high_epoch_d_model_name = 'd_high_epoch'
    g_model_save_path = './models/' + high_epoch_g_model_name
    d_model_save_path = './models/' + high_epoch_d_model_name


    dataloader = load_data_train(opt)

    model = Fine_Tune_Disc(opt, dataloader)
    model.cuda()

    model_folder_path = './models/'
    load_model_epoch = ['epoch_{0}'.format(low_epoch), 'epoch_{0}'.format(high_epoch)]
    model.train(load_model_epoch, model_folder_path)

    print("stage 2 result:")
    high_epoch_g_model_name = 'g_high_epoch2'
    high_epoch_d_model_name = 'd_high_epoch2'
