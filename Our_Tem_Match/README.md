# Anomaly detection method based on fast template matching
We introduce an efficient template matching method, which fully utilizes template information to improve performance. 
<div align=center><img src="https://github.com/PPTdataset/PPT/blob/master/imgs/model.jpg" width="800"></div>

## Usage

### Environment
You can create an anaconda environment and install.
```bash
conda create -n template python=3.7
conda activate template
pip install -r requirements.txt
```
### Datasets
Download the [dataset](https://drive.google.com/drive/folders/1GKFCRwqyEC8j5c8mjWWjk_Se5c6lUNvn) image and put them under `raw_data` and `val_data` respectively.

You should put train and test data under `raw_data` and have the following directory & file structure:
```
Raw_data
├── Template
│   └── Image_1.bmp
│   └── Image_2.bmp
│   ...
│   └── Image_100.bmp
├── OK_img
│   └── 0a0dbecdc0b141618c0624406de9b3c3.bmp
│   ...
│   └── ffe350fb4000413cb26784673e1e52b2.bmp
├── TC_img
│   └── 000ZYVs38Gj08xgQ3yC1el0r4fv8J6.bmp
│   ...
│   └── ZzYLP01EkT1uySNYQO81w3g1mZ5u1b.bmp
```

### Model
Our model uses a pre-trained ResNet network as the initial value, so it requires downloading the pre-trained resnet [model](https://drive.google.com/drive/folders/16QjOt0Y1UoD4pYafKfJKM6XH37m9pwLx) and put it under `temp_data/pretrained_model`.

If you want to view the results directly, place the pre-trained template matching [model](https://drive.google.com/drive/folders/16QjOt0Y1UoD4pYafKfJKM6XH37m9pwLx) in `model`. See the following directory structure for an example:
```
model
├── checkpoints
│   ├──Template_Matching_seamless_clone
│   │   └──  latest_net_G.pth
├── TC_property
│   ├──Template_Matching
│   │   └──  Template_Matching_r4
```

### Train and Test:
To train and test, run
```bash
cd main_code
python main.py
```
You can change the options in `main_code/options/main_options.py`



