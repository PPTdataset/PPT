# -*- coding: utf-8 -*-
ROOT = '..'

#################################################

def base_parse(opt):
    opt.name = 'Template Matching'
    opt.input_nc = 6 
    opt.output_nc = 1
    opt.model_name = 'seg_plus'
    opt.input_size = 128
    return opt


#################################################


def train_parse_base(opt, use_plus=True):
    opt.topk = 8 #8 16 24
    opt.n_epochs = 1
    opt.contrast_thr = 10 #thr
    opt.resize_ratio = 4
    opt.dice_loss_eps = 1.0
    opt.no_template = 0 #consistent with the input channel!
    opt.simple_defect = 0
    if use_plus:
        opt.no_defect_rate = 0.35
        opt.use_same_tc = 0.6666666
        opt.dataset_mode = opt.model_name
        # opt.dataset_mode = 'seg_v10_plus_ok'
    opt.model = opt.model_name
    opt.netG = opt.model_name
    # opt.num_residual_layer = [2, 3, 4]
    opt.contrast_thr_low = 5
    opt.mask_ignore = 0
    opt.batch_size = 10  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 1/10 : 10
    opt.lr = 0.001
    opt.defect_generator = 'seamless_clone'
    opt.lambda_loss_L1 = 1.
    opt.data_augmentation = 0
    opt.is_sigmoid = 1
    opt.region_thr = 10
    opt.num_threads = 0
    opt.dataroot = '%s/raw_data/' % ROOT
    opt.checkpoints_dir = '%s/model/checkpoints/' % ROOT
    opt.no_html = True
    opt.no_flip = True
    opt.lr_policy = 'cosine'
    opt.print_freq = 100
    opt.check_img_freq = 1000
    opt.save_epoch_freq = 1

    opt.print_log = 1  # must be 0
    opt.save_img = 0  # must be 0
    opt.save_check_img = 1  # must be 0

    return opt


#################################################


def test_parse_base(opt):
    opt.no_template = 0
    opt.resize_ratio = 4
    # opt.num_residual_layer = [2, 3, 4]
    opt.dataset_mode = 'test_'+opt.model_name
    opt.model = 'test_'+opt.model_name
    opt.netG = opt.model_name
    opt.save_images_versoin = 'save_images_seg_v9'
    opt.defect_generator = 'seamless_clone'
    opt.is_sigmoid = 1
    opt.sigmoid_thr = 0.5
    opt.batch_size = 80
    opt.num_threads = 0
    opt.eval = True

    opt.print_log = 1  
    opt.load_model_from_models = 0  
    opt.save_img = 1  
    opt.is_val = 1  
    #if opt.is_val:
    #    opt.dataset_mode = 'test_seg_v10_val'
    opt.dataroot = '%s/raw_data/' % ROOT
    #opt.val_pre_path = '%s/temp_data/results_validation/' % ROOT
    opt.checkpoints_dir = '%s/model/checkpoints/' % ROOT
    opt.val_gt_path = '%s/val_data/' % ROOT
    opt.results_dir = '%s/temp_data/results_validation/' % ROOT
    return opt
