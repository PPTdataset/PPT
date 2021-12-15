# PPT
We share the specific code implementation of the following method. For some of the latest methods, such as KD, we directly use the source code given in the article. For some of the more classic Baseline methods, such as AutoEncoder, we implemented it ourselves.

In code implementation, we make some minor adjustments and choose the best hyperparameters to achieve the goal of anomaly detection in our PPT dataset. The principle is not adding additional tricks or modifying the network structure if possible.

The [dataset](https://drive.google.com/drive/folders/1GKFCRwqyEC8j5c8mjWWjk_Se5c6lUNvn) image and pretrained [model](https://drive.google.com/drive/folders/16QjOt0Y1UoD4pYafKfJKM6XH37m9pwLx) can be downloaded by clicking the link.

<div align=center><img src="https://github.com/PPTdataset/PPT/blob/master/imgs/abstract.jpg" width="600"></div>

For the specific operation of the evaluation method code, please refer to the readme in the corresponding folder.

<div align=center><img src="https://github.com/PPTdataset/PPT/blob/master/imgs/eva.jpg" width="800"></div>

## References
<span id="OG">[1] Zaheer, Muhammad Zaigham, et al. "Old is gold: Redefining the adversarially learned one-class classifier training paradigm." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.  </span>  
<span id="Cycle">[2] Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017.  </span>  
<span id="KD">[3] Salehi, Mohammadreza, et al. "Multiresolution knowledge distillation for anomaly detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.  </span>  
