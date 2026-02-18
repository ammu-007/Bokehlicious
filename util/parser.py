from argparse import ArgumentParser
from pathlib import Path

def get_base_parser():
    parser = ArgumentParser()
    parser.add_argument('-out_path', type=Path, default='./output', help='output folder, default is \'./output\'')
    parser.add_argument('-image_format', type=str, default='png', choices=['png', 'jpg'], help='image format for saving outputs')
    return parser

def add_network_args(parser: ArgumentParser):
    parser.add_argument('-size', type=str, required=True, choices=['small', 'large', 'defocus_deblur'])
    parser.add_argument('-device', type=str, default='cuda', choices=['cuda', 'cpu'])

def add_img_args(parser: ArgumentParser):
    parser.add_argument('-img_path', type=str, required=True)
    parser.add_argument('-av', type=float, default=2.0)
    parser.add_argument('-max_dim', type=int, default=2000)
    parser.add_argument('-min_divisor', type=int, default=4)

def get_predict_parser():
    parser = get_base_parser()
    add_network_args(parser)
    add_img_args(parser)
    return parser

def get_eval_parser():
    parser = get_base_parser()
    add_network_args(parser)
    parser.add_argument('-dataset', type=str, required=True, choices=['RealBokeh', 'RealBokeh_bin', 'EBB400', 'EBB_Val294'])
    parser.add_argument('--save_outputs', action='store_true')
    return parser

def get_train_parser():
    parser = ArgumentParser(description='Train Bokehlicious on the RealBokeh dataset.')
    # Model
    parser.add_argument('-size', type=str, required=True, choices=['small', 'large'],
                        help='Model size to train')
    parser.add_argument('-device', type=str, default='cuda', choices=['cuda', 'cpu'])
    # Data
    parser.add_argument('-data_path', type=Path, default='./dataset/RealBokeh_3MP',
                        help='Path to RealBokeh_3MP directory (must contain train/ and validation/ subdirs)')
    parser.add_argument('-num_workers', type=int, default=4,
                        help='DataLoader worker processes (default: 4, use 0 for debugging)')
    # Training hyperparameters (paper: batch=4, patch=512, lr=5e-4, lambda=0.6)
    parser.add_argument('-epochs', type=int, default=200)
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-patch_size', type=int, default=512,
                        help='Random crop size for training patches (paper: 512)')
    parser.add_argument('-lr', type=float, default=5e-4,
                        help='Adam learning rate (paper: 5e-4)')
    parser.add_argument('-lambda_lpips', type=float, default=0.6,
                        help='LPIPS loss weight (paper Eq. 5: lambda=0.6)')
    # Checkpointing & logging
    parser.add_argument('-checkpoint_dir', type=Path, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('-resume', type=Path, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('-log_dir', type=Path, default='./runs',
                        help='TensorBoard log directory')
    parser.add_argument('-val_freq', type=int, default=5,
                        help='Validate every N epochs')
    parser.add_argument('-save_freq', type=int, default=10,
                        help='Save a periodic checkpoint every N epochs')
    return parser

def get_ntire_parser():
    parser = get_base_parser()
    parser.description=(
        "This script produces a submission ready .zip archive to be uploaded at "
        "https://www.codabench.org/competitions/12764/#/participate-tab for evaluation by our server. \n"
        "Use \'-n [NAME]\' to set name of your architecture in the leaderboard.\n"
        "Put the development (and the final test input= image archive to the \'./dataset\' folder.\n"
        "When the test phase starts, use \'-p test\' to load the test set.")
    parser.add_argument('-name', '-n', type=str, required=True,
                        help='name of your method to be shown on the leaderboard')
    parser.add_argument('-phase', '-p', type=str, choices=['dev', 'test'], default='dev',
                        help='current phase of the challenge')
    parser.add_argument('-checkpoint', '-c', type=str, required=True,
                        help='name of your checkpoint in the \'./checkpoint\' folder\'.')
    parser.add_argument('--extra_data', '--ed', action='store_true',
                        help='activate if data other than RealBokeh was used to train your model')
    parser.add_argument('-device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='device to use')
    parser.add_argument('-dataset_root_dir', '-dr', type=Path, default='./dataset',
                        help='path to the dir containing the Bokeh_NTIRE dataset folder/archive, default is \'./dataset\'')
    return parser