from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options

        parser.add_argument('--val_gt_path', default="val_gt_path", type=str, help='path to gt json file for val set')
        parser.add_argument('--results_dir', default="results_dir", type=str, help='path to submit json file for TC_img')
        parser.add_argument('--is_val', default=0, type=int, help='path to submit json file for TC_img')
        parser.add_argument('--save_json', type=int, default=1, help='1: save_json or 0: save mask')

        parser.add_argument('--val_pre_path', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        # parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=float("inf"), help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
