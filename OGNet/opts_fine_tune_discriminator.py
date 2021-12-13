import argparse


# Phase Two Related options
def parse_opts_ft():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/train/', type=str, help='Input file path')
    parser.add_argument('--normal_class', default='0', type=str, help='normal_class')
    parser.add_argument('--d_learning_rate', default='0.00005', type=float, help='d_learning_rate')
    parser.add_argument('--high_epoch_fake_loss_contribution', default='0.9', type=float, help='the contribution of Gn fake loss being added compared with the loss of real images in total_normal_loss')
    parser.add_argument('--psuedo_anomaly_contribution', default='0.95', type=float, help='the contribution of pseudo anomalies loss being added compared with the loss of Go reconstructed images in total_normal_loss')
    parser.add_argument('--epoch', default=20, type=int, help='Epoch for training')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--n_threads', default=8, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--batch_shuffle', default=True, type=bool, help='shuffle input batch or not')
    parser.add_argument('--test_anomaly_threshold', default='0.5', type=float, help='threshold to print anomaly or not')
    parser.add_argument('--drop_last', default=True, type=bool, help='drop the remaining of the batch if the size doesnt match minimum batch size')
    parser.add_argument('--frame_size', default=45, type=int, help='one side size of the square patch to be extracted from each frame')
    parser.add_argument('--low_epoch', default=10, type=int, help='low epoch i.e. Gold for phase two')
    parser.add_argument('--high_epoch', default=49, type=int, help='high epoch i.e. Gn for phase two')
    parser.add_argument('--iterations', default=75, type=int, help='iterations for phase two')
    args = parser.parse_args()
    return args

