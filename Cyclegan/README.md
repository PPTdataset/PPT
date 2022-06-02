# CycleGAN

We provide PyTorch implementations for both unpaired and paired image-to-image translation.

The origin code was written by [Jun-Yan Zhu](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We made some changes to the PPT dataset on this origin code.


## Getting Started
- Download PPT dataset from([dataset](https://drive.google.com/drive/folders/1GKFCRwqyEC8j5c8mjWWjk_Se5c6lUNvn?usp=sharing))
Put 'OK_img' and 'DF_img' in 'raw_data/round_train'

- Train a model:
```bash
bash train_cyclegan.sh
```
To see more intermediate results, check out `temp_data/checkpoints/`.

- Test the model:
```bash
bash train_cyclegan.sh
```
- The test results will be saved in `saves`.
