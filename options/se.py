import argparse
from pathlib import Path


def add_model_options(parser: argparse.ArgumentParser):
    

    # transformer
    parser.add_argument('--feature_dim', type=int, default=512, help='dimension of the hidden feature')
    parser.add_argument('--n_heads', type=int, default=16, help='number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6, help='number of encoder/decoder layers')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='ratio of the hidden dimension of the MLP')

    # sequence
    parser.add_argument('--n_motions', type=int, default=120, help='number of motions in a sequence')
    parser.add_argument('--fps', type=int, default=30, help='frame per second')


def add_data_options(parser: argparse.ArgumentParser):
    
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')


def add_training_options(parser: argparse.ArgumentParser):
    parser.add_argument('--exp_name', type=str, default='HDTF_TFHP', help='experiment name')
    parser.add_argument('--max_iter', type=int, default=100000, help='max number of iterations')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature for contrastive loss')

    parser.add_argument('--save_iter', type=int, default=1000, help='save model every x iterations')
    parser.add_argument('--val_iter', type=int, default=2000, help='validate every x iterations')
    parser.add_argument('--log_iter', type=int, default=50, help='log to tensorboard every x iterations')
    parser.add_argument('--log_smooth_win', type=int, default=50, help='smooth window for logging')
