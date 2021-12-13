python3 test.py \
    --name cycle_gan \
    --dataroot ../raw_data/ \
    --load_model_dir ../temp_data/checkpoints/cycle_gan/seamless_clone_no_rule_v1/latest_net_G.pth \
    --output_path ../temp_data/result/data/cycle_gan/TC_Images/ \
    --val_gt_path ../val_data/json_PPT \
    --if_res 1 \
    --contrast_thr 25 \
    --use_mask 1 \
    --batch_size 1 \
    --netG resnet_9blocks \
    --norm batch \
    --ngf 128 \
    --model cycle_gan \
    --direction BtoA \
    --dataset_mode test_cyclegan \
    --checkpoints_dir ../temp_data/checkpoints/ \
    --results_dir ../temp_data/result_val/ \
    --save_img 1 \
    --load_model_from_models 0 \
    --is_val 0 \
    --input_nc 3 \
    --output_nc 3 \
    --model_suffix _A\