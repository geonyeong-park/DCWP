import os
import argparse
from copy import deepcopy

from munch import Munch
from torch.backends import cudnn
import torch

from data.data_loader import get_original_loader, get_val_loader, \
    get_aug_loader, get_concat_loader
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
        loaders = Munch(unsup=get_original_loader(args, mode='unsup'),
                        sup=get_original_loader(args, mode='sup'),
                        concat=get_concat_loader(args),
                        val=get_val_loader(args))

        solver.augment(loaders)

    elif args.mode == 'debias':
        loaders = Munch(unsup=get_aug_loader(args, mode='unsup'),
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



    parser.add_argument('--dim_in', type=int, default=32)




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

    # training arguments
    parser.add_argument('--bias_total_iters', type=int, default=20000,
                        help='Number of training iterations for biasing model')
    parser.add_argument('--vae_total_iters', type=int, default=20000,
                        help='Number of training iterations for training vae')
    parser.add_argument('--bias_resume_iter', type=int, default=0,
                        help='Iterations to resume biased model')
    parser.add_argument('--vae_resume_iter', type=int, default=0,
                        help='Iterations to resume VAE')
    parser.add_argument('--attack_iters', type=int, default=7)
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate in util/params.py will be used')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Not used')
    parser.add_argument('--lr_decay_step', type=int, default=10000)
    parser.add_argument('--lr_gamma', type=float, default=0.5)

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
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=100)

    args = parser.parse_args()
    main(args)
