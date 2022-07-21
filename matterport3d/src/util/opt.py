import argparse
class Options:
    def __init__(self, prog=None):
        # =====================
        # Image Detail
        # =====================
        self.imw = 1024
        self.imh = 256
        self.resize = False
        self.resize_scale = 2

        # =====================
        # Dataset
        # =====================
        self.workers = 0
        self.equi_coord = '..\\data\\pano_coord_1024.npy'
        self.equi = True

        # Training Param #
        # self.train_path = '\\home\\juliussurya\\work\\cat2dog\\train_b.txt'
        self.train_path = 'D:\\work\\Data\\3d60_dataset_rgb_train'
        self.train_path_depth = 'D:\\work\\Data\\3d60_dataset_depth_train'
        self.train_batch = 2
        self.train_shuffle = False
        self.train_len = 6947
        self.train_len_depth = 6947
        self.ckpt_path = '..\\ckpt'
        self.model_path = '.\\trained_model'
        self.learn_rate = 0.0002
        self.lr_d = 0.0005
        self.lr_g = 0.0001
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.total_epochs = 77
        self.train_log = '..\\log'
        self.train_log_depth = '..\\log_depth'
        self.print_step = 500
        self.faed_gt_activations_rgb = '.\\mt3d_rgb_500.npy'
        self.faed_gt_activations_depth = '.\\mt3d_depth_500.npy'
        self.pretrained_model_path_rgb = '.\\trained_model\\matterport3d_fusion_1_rgb_212_77'
        self.pretrained_model_path_depth = '.\\trained_model\\matterport3d_fusion_1_depth_212_77'
        self.faed_data_len=500
        self.faed_data_offset=0


        # Test Param #
        self.test_path = 'D:\\work\\Data\\3d60_dataset_rgb_train'
        self.test_path_depth = 'D:\\work\\Data\\3d60_dataset_rgb_train'
        self.test_batch = 1
        self.test_shuffle = False
        self.test_len = 360#5000
        self.output_path = '..\\output'

        # Val param #
        self.val_path = '.\\pano_data_val'
        self.val_batch = 1
        self.val_shuffle = True
        self.val_len = 360#5000

        self.net = None

        self.parser = argparse.ArgumentParser(prog=prog)
        self._define_parser()
        self._parse()
        # =====================

    def _define_parser(self):
        self.parser.add_argument('--net', default='small',
                                 help='Network type (small || medium || large')

    def _parse(self):
        args = self.parser.parse_args()
        self.net = args.net

