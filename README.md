# PPT: ANOMALY DETECTION DATASET OF PRINTED PRODUCTS WITH TEMPLATES
## Abstract
<div align=center><img src="https://github.com/PPTdataset/PPT/blob/master/imgs/abstract.jpg" width="500"></div>
Visual anomaly detection has been an active topic in industrial applications. In particular, it aims to classify anomalies and precisely locate defective areas in the printed products. To the best of our knowledge, there is no anomaly detection dataset for industrial printings. In this project, we are the first to introduce a [Printed Products with Templates (PPT) dataset](https://drive.google.com/drive/folders/1GKFCRwqyEC8j5c8mjWWjk_Se5c6lUNvn), which contains large templates and sliced images collected from industry scene images. PPT is a challenging dataset with more variable surface defects and more disturbing background than existing related benchmarks. Furthermore, we propose a template matching method for anomaly detection of printed products, which consists of a fast template matching block with a convolutional operation using the test sliced image as its kernel, and a prediction network for generating an anomaly map of the test sliced image. Experimental results show that our method achieves state-of-the-art performance compared to the related anomaly detection approaches.

<div align=center><img src="https://github.com/PPTdataset/PPT/blob/master/imgs/eva.jpg" width="800"></div>

## Description
We share the specific code implementation of the evaluated methods. This repository documents the code for these evaluation methods for the convenience of the community to reproduce. 

If you want to reproduce our proposed template matching anomaly detection method only, please check the contents of the `Our_Tem_Match` folder.

For other evaluated methods, we directly use the source code of these methods and make some minor adjustments to achieve better results in our PPT dataset. The principle is not adding additional tricks or modifying the network structure if possible.

If you want to reproduce these methods on our dataset, please refer to the readme in the corresponding folder.

## References
<span id="OG">[1] Zaheer, Muhammad Zaigham, et al. "Old is gold: Redefining the adversarially learned one-class classifier training paradigm." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.  </span>  
<span id="Cycle">[2] Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017.  </span>  
<span id="KD">[3] Salehi, Mohammadreza, et al. "Multiresolution knowledge distillation for anomaly detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.  </span>  
<span id="ganomaly">[4] Akcay, Samet, Amir Atapour-Abarghouei, and Toby P. Breckon. "Ganomaly: Semi-supervised anomaly detection via adversarial training." Asian conference on computer vision. Springer, Cham, 2018.  </span>  
<span id="MNAD">[5] Park, Hyunjong, Jongyoun Noh, and Bumsub Ham. "Learning memory-guided normality for anomaly detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.  </span>  
