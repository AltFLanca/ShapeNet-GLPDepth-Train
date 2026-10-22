import argparse


class BaseOptions():
    def __init__(self):
        pass

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--gpu_or_cpu', type=str, default='gpu')
        parser.add_argument('--data_path', type=str, required=True,
                            help='Path to the shapenetsem directory')
        parser.add_argument('--dataset', type=str, default='shapenetsem_normalized_corrected',
                            choices=['shapenetsem', 'shapenetsem_normalized', 'shapenetsem_normalized_corrected'])
        parser.add_argument('--exp_name', type=str, default='test')
        parser.add_argument('--batch_size', type=int, default=3)
        parser.add_argument('--workers', type=int, default=1)

        parser.add_argument('--train_txt_path', type=str, default=None)
        parser.add_argument('--test_txt_path', type=str, default=None)
        parser.add_argument('--metadata_path', type=str, default=None)

        parser.add_argument('--max_depth', type=float, default=10.0)
        parser.add_argument('--max_depth_eval', type=float, default=10.0)
        parser.add_argument('--min_depth_eval', type=float, default=1e-3)

        return parser
