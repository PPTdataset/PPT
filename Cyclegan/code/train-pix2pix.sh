python3 train.py \
    --name pix2pix \
    --defect_generator seamless_clone \
    --version no_rule_v1 \
    --dataroot ../raw_data/ \
    --lambda_loss_L1 1.5 \
    --dataset_mode aligned \
    --data_augmentation 4 \
    --save_img 1 \
    --use_mask 0 \
    --is_sigmoid 0 \
    --contrast_thr 15 \
    --model pix2pix \
    --direction BtoA \
    --display_id 0 \
    --num_threads 0 \
    --checkpoints_dir ../temp_data/checkpoints/ \
    --gpu_ids 0 \
    --input_nc 3 \
    --output_nc 3 \
    --no_html \
    --batch_size 1 \
    --netG unet_256 \
    --norm batch \
    --ngf 128 \
    --netD basic \
    --gan_mode vanilla \
    --n_epochs 200 \
    --n_epochs_decay 200 