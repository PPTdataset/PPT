from PIL import Image
import glob
import os

out_dir = 'Dataset/PPT/5k/ground_truth/subnew/'
in_dir ='Dataset/PPT/5k/ground_truth/sub/'
cnt = 0
for img in os.listdir(in_dir):
    Image.open(os.path.join(in_dir,img)).save(os.path.join(os.path.join(out_dir,img[:-4])+ '.png'))
    