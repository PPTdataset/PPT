
from model import OGNet
from opts import parse_opts
from dataloader import load_data_train

# Code for the CVPR 2020 paper ->  Old is Gold: Redefining the Adversarially Learned One-Class Classifier Training Paradigm
# https://arxiv.org/abs/2004.07657
# http://openaccess.thecvf.com/content_CVPR_2020/html/Zaheer_Old_Is_Gold_Redefining_the_Adversarially_Learned_One-Class_Classifier_Training_CVPR_2020_paper.html
# If you use this code, or find it helpful, please cite our paper:
#  @inproceedings{zaheer2020old,
#    title={Old is Gold: Redefining the Adversarially Learned One-Class Classifier Training Paradigm},
#    author={Zaheer, Muhammad Zaigham and Lee, Jin-ha and Astrid, Marcella and Lee, Seung-Ik},
#    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#    pages={14183--14193},
#    year={2020}
#  }
#
# Please contact me through email, if you have any questions or need any help: mzz . pieas (at) gmailcom
if __name__=="__main__":

    opt = parse_opts()
    train_loader = load_data_train(opt)
    model = OGNet(opt, train_loader)
    model.cuda()
    model.train(opt.normal_class)

