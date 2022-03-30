import os
import argparse
from copy import deepcopy

from munch import Munch
from torch.backends import cudnn
import torch

from data.data_loader import get_original_loader, get_val_loader, \
    get_aug_loader
from training.augment_solver import AugmentSolver
#from training.debias_solver import DebiasingSolver
#from training.test_solver import TestSolver
from util import setup, save_config


def main(args):
    print(args)
    args = setup(args) # Making folders following exp_name
    save_config(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    if args.mode == 'augment':
        solver = AugmentSolver(args)
    elif args.mode == 'debias':
        solver = DebiasingSolver(args)
    elif args.mode == 'test':
        solver = TestSolver(args)
    else:
        raise NotImplementedError

    if args.mode == 'augment':
        loaders = Munch(unsup=get_original_loader(args, mode='unsup'), # Could be None
                        sup=get_original_loader(args, mode='sup'),
                        sup_dataset=get_original_loader(args, mode='sup', return_dataset=True),
                        val=get_val_loader(args))

        solver.augment(loaders)

    elif args.mode == 'debias':
        loaders = Munch(unsup=get_aug_loader(args, mode='unsup'), # Could be None
                        sup=get_aug_loader(args, mode='sup'),
                        val=get_val_loader(args))

        solver.debias(loaders)

    elif args.mode == 'test':
        loaders = Munch(val=get_val_loader(args))

        solver.evaluate(loaders)

        print_scores(args, solver.metrics)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data arguments
    parser.add_argument('--data', type=str, default='cmnist',
                        choices=['cmnist', 'cifar10c', 'bffhq'])
    parser.add_argument('--labeled_ratio', type=float, default=0.1, choices=[0.01,0.1,1.],
                        help='Ratio of labeled data')
    parser.add_argument('--use_unsup_data', default=False, action='store_true',
                        help='If labeled_ratio < 1, use_unsup_data=True. Otherwise False. See util.__init__')
    parser.add_argument('--conflict_pct', type=float, default=1., choices=[0.5,1.,2.,5.],
                        help='Percent of bias-conflicting data')


    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1)
    parser.add_argument('--lambda_bias', type=float, default=1)
    parser.add_argument('--lambda_debias', type=float, default=1)
    parser.add_argument('--lambda_cyc', type=float, default=10)

    # training arguments
    parser.add_argument('--GAN_total_iters', type=int, default=100000,
                        help='Number of training iterations for training GAN')
    parser.add_argument('--bias_resume_iter', type=int, default=0,
                        help='Iterations to resume biased model, NOT used')
    parser.add_argument('--GAN_resume_iter', type=int, default=0,
                        help='Iterations to resume GAN')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--lr_G', type=float, default=1e-4,
                        help='lr for G, E')
    parser.add_argument('--lr_D', type=float, default=5e-4,
                        help='lr for D')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Not used')
    parser.add_argument('--lr_decay_step', type=int, default=10000,
                        help='Not used when training GAN')
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--dim_in', type=int, default=32,
                        help='number of channels in first hidden layer of G')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='dimension of final style code')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='dimension of initial 2D domain code')
    parser.add_argument('--g_every', type=int, default=1)

    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['augment', 'debias', 'test'],
                        help='''This argument is used in solver.
                        augment: pre-train or load VAE+biased_model. Do adversarial augmentation.
                        debias: Debias main model with augmented images.
                        test: evaluate performance''')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=7777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_root_dir', type=str, default='/home/pky/research/unsup_dataset')
    parser.add_argument('--val_root_dir', type=str, default='/home/pky/research/dataset')
    parser.add_argument('--log_dir', type=str, default='expr/log')
    parser.add_argument('--result_dir', type=str, default='expr/results',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Nametag for the experiment')

    # step size
    parser.add_argument('--print_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=1000)

    args = parser.parse_args()
    main(args)
