from argparse import ArgumentParser
from pathlib import Path

def get_predict_parser():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, default='./configs/default.yaml',
                        help='Path to the YAML configuration file for the experiment.')
    return parser

def get_eval_parser():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, default='./configs/default.yaml',
                        help='Path to the YAML configuration file for the experiment.')
    return parser

def get_train_parser():
    parser = ArgumentParser(description='Train Bokehlicious on the RealBokeh dataset.')
    parser.add_argument('-c', '--config', type=Path, default='./configs/default.yaml',
                        help='Path to the YAML configuration file for the experiment.')
    return parser

def get_ntire_parser():
    parser = ArgumentParser()
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