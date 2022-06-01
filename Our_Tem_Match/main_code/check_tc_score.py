import pickle
import os
import torch
import cv2
from data.image_folder import make_dataset
from tools.utils import Combine_img, Save, Make_dirs

if __name__ == '__main__':
    ROOT = '.'
    SAVEIMG = "%s/temp_data/saves" %ROOT
    if os.path.exists(SAVEIMG):
        import shutil
        shutil.rmtree(SAVEIMG)
    Make_dirs(SAVEIMG)
    save_dir = './model/TC_property/part3_r8_c3' 
    dir_OK = './raw_data/round_test/part3/OK_Images/'
    dir_TC = './raw_data/round_test/part3/TC_Images/'
    # self.dir_TC = './debug/'

    # load images
    OK_paths = sorted(make_dataset(dir_OK))
    TC_paths = sorted(make_dataset(dir_TC))

    f = open(save_dir,'rb')  
    TC_property = pickle.load(f)
    TC_position = torch.tensor(TC_property['TC_position'])
    TC_position_score = torch.tensor(TC_property['TC_position_score'])
    TC_valid_keypoints = torch.tensor(TC_property['TC_valid_keypoints'])
    f.close()

    check_num_tc = 100
    check_num_ok = 10
    values, indices = TC_position_score.topk(check_num_ok, dim=0)
    indices_top1 = indices[0,:].squeeze()

    values_tc, indices_tc = TC_position_score[indices_top1, range(len(indices_top1))].topk(check_num_tc, largest=False)

    OK_img_list = {}
    for rank_tc in range(check_num_tc):
        index_tc = indices_tc[rank_tc]
        TC_img = cv2.imread(TC_paths[index_tc])
        img_list = [TC_img]
        for rank in range(check_num_ok):
            score = values[rank][index_tc]
            index_ok = indices[rank][index_tc]
            position = TC_position[index_ok][index_tc]
            if index_ok in OK_img_list:
                OK_img = OK_img_list[index_ok]
            else:
                OK_img = cv2.imread(OK_paths[index_ok])
                OK_img_list[index_ok] = OK_img
            
            OK_img = OK_img[position[0]: position[0] + 128,
                            position[1]: position[1] + 128]
            cv2.putText(OK_img, '%.5f_%d' %(score, index_ok), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            img_list.append(OK_img)
        Save(Combine_img(3, img_list), '%03d_%d' %(rank_tc, index_tc))
        print('save', index_tc)