import argparse


def setting_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_checkpoint_dir', default='ckpt',
                        help='Path to checkpoint file.')

    parser.add_argument('--save_top_k', default=30,
                        help='save_top_k for train.')

    parser.add_argument('--gpus', type=str, default='0,1',
                        help='gpus for train.')

    parser.add_argument('--n_max_epochs', type=int, default=200,
                        help='max_epochs for train.')

    parser.add_argument('--batch_sizes', type=int, default=16,
                        help='batch_sizes for train.')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')

    return parser.parse_args()
