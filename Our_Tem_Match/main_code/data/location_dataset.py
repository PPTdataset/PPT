from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import cv2
from tools.defect_generator import direct_add, seamless_clone, ok2df_without_tc
from tools.utils import Random, Save, FixCrop, get_transform_location



class LocationDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths

        # self.dir_OK = os.path.join(opt.dataroot, 'OK_Images/')
        # self.OK_paths = sorted(make_dataset(self.dir_OK, opt.max_dataset_size))
        # self.OK_size = len(self.OK_paths)  # get the size of dataset A

        self.OK_list = opt.checkpoint['part%s' %opt.dataroot[-2]]
        self.OK_size = len(self.OK_list)

        # self.OK_size = 1
        self.grayscale = opt.output_nc == 1

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        # index = random.randint(0, 4)

        # OK_path = self.OK_paths[index]
        # OK_img = Image.open(OK_path).convert('RGB')
        
        OK_img = self.OK_list[index]        
        # OK_img = self.OK_list[0]        

        OK_img2 = OK_img.copy()

        OK_img = Image.fromarray(cv2.cvtColor(OK_img, cv2.COLOR_BGR2RGB))        # OpenCV to PIL.Image

        w, h = OK_img.size
        self.opt.load_size_loc = [int(h / self.opt.resize_ratio), int(w / self.opt.resize_ratio)]

        data_transform = get_transform_location(self.opt, grayscale=self.grayscale, if_resize = True)
        OK_img1 = data_transform(OK_img)

        data_transform_ori = get_transform_location(self.opt, grayscale=self.grayscale, if_resize = False)
        OK_img2 = data_transform_ori(OK_img)
        return OK_img1, OK_img2

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.OK_size
