# Knowledge distillation
The origin code was written by [Salehi](https://github.com/Niousha12/Knowledge_Distillation_AD). We made some changes to the PPT dataset on this origin code.



## Datsets:
- Download datasets [here](https://drive.google.com/drive/folders/1GKFCRwqyEC8j5c8mjWWjk_Se5c6lUNvn?usp=sharing) and place the training, testing, and ground images in the subdirectories of 'Dataset/PPT/5K'

##### For Localization test you should remove the `good` folder in `5k/test/` folder.

## Train the Model:
Start the training using the following command. The checkpoints will be saved in the folder `outputs/{experiment_name}/{dataset_name}/checkpoints`.

Train parameters such as experiment_name, dataset_name, normal_class, batch_size and etc. can be specified in `configs/config.yaml`.
``` bash
python train.py --config configs/config.yaml
```

## Test the Trained Model:
You can directly run the test.py by downloading the pre-trained model from [here](https://drive.google.com/drive/folders/16QjOt0Y1UoD4pYafKfJKM6XH37m9pwLx?usp=sharing)and place inside the directory `outputs/{experiment_name}/{dataset_name}/checkpoints`
Test parameters can also be specified in `configs/config.yaml`.
``` bash
python test.py --config configs/config.yaml
```
