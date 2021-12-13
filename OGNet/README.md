The origin code was written by [Zaheer](https://github.com/xaggi/OGNet). We made some changes to the PPT dataset on this origin code.

## Requirements

- Python2.7
- torch 1.2.0
- torchvision 0.4.0

## Code execution

- Train.py is the entry point to the code.
- Download datasets [here](https://drive.google.com/drive/folders/1GKFCRwqyEC8j5c8mjWWjk_Se5c6lUNvn?usp=sharing) and place train images in the subdirectories of ./data/train/
  - Example:
    - All images from inlier class (\*.png) should be placed as ./data/train/0/sub/*.png
    - Similarly, all images from outlier class (* \*.png) should be placed as ./data/train/1/sub/* \*.png
- Set necessary options in opts.py for phase one and opts_fine_tune_discriminator.py for phase two.
- Execute Train.py


- You can directly run the test.py by downloading the pre-trained model
- Download trained generator and discriminator models from [here](https://drive.google.com/drive/folders/16QjOt0Y1UoD4pYafKfJKM6XH37m9pwLx?usp=sharing) and place inside the directory ./models/

- run test.py


