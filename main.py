import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch

from training.pruning_solver import PruneSolver
from util import setup, save_config, modify_args_for_baselines


def main(args):
    print(args)
    args = setup(args) # Making folders following exp_name
    save_config(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = PruneSolver(args)

    if args.phase == 'train':
        solver.train()
    else:
        solver.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, required=True,
                        choices=['prune', 'MRM'])

    # Data arguments
    parser.add_argument('--data', type=str, default='cmnist',
                        choices=['cmnist', 'cifar10c', 'bffhq', 'celebA'])
    parser.add_argument('--cmnist_use_mlp', default=False, action='store_true')
    parser.add_argument('--conflict_pct', type=float, default=5., choices=[0.5, 1., 2., 5.],
                        help='Percent of bias-conflicting data')
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'])

    # weight for objective functions
    parser.add_argument('--lambda_con_prune', type=float, default=0.05)
    parser.add_argument('--lambda_con_retrain', type=float, default=0.05)
    parser.add_argument('--lambda_sparse', type=float, default=1e-8)
    parser.add_argument('--lambda_upweight', type=float, default=20)

    # training arguments
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--no_lr_scheduling', default=False, action='store_true')

    # Pretraining
    parser.add_argument('--lr_decay_step_pre', type=int, default=600)
    parser.add_argument('--lr_gamma_pre', type=float, default=0.1)
    parser.add_argument('--lr_pre', type=float, default=1e-1)
    parser.add_argument('--pretrain_iter', type=int, default=2000)

    # Pruning
    parser.add_argument('--lr_prune', type=float, default=1e-2)
    parser.add_argument('--pruning_iter', type=int, default=2000)
    parser.add_argument('--earlystop_iter', type=int, default=None)

    # Retraining
    parser.add_argument('--optimizer', type=str, required=False,
                        choices=['Adam', 'SGD'], default='Adam')
    parser.add_argument('--lr_main', type=float, default=1e-2)
    parser.add_argument('--retrain_iter', type=int, default=500)
    parser.add_argument('--lr_decay_step_main', type=int, default=600)
    parser.add_argument('--lr_gamma_main', type=float, default=0.1)

    parser.add_argument('--weight_decay', type=float, default=1e-4) #TODO: weight decay is important in JTT!
    parser.add_argument('--reinitialize', default=False, action='store_true') # MRM
    parser.add_argument('--uniform_weight', default=False, action='store_true') # MRM
    parser.add_argument('--select_with_GCE', default=False, action='store_true')

    # For FeatureSwap
    parser.add_argument('--total_iter', type=int, default=20000)
    parser.add_argument('--swap_iter', type=int, default=10000)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.99)
    parser.add_argument('--lambda_swap', type=float, default=1)
    parser.add_argument('--lambda_dis_align', type=float, default=10)
    parser.add_argument('--lambda_swap_align', type=float, default=10)

    # misc
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=7777,
                        help='Seed for random number generator')
    parser.add_argument('--imagenet', default=True, action='store_true')
    parser.add_argument('--supervised', default=False, action='store_true',
                        help='Use true bias label or not')
    parser.add_argument('--pseudo_label_method', type=str, required=False,
                        choices=['wrong', 'ensemble'], default='ensemble')
    parser.add_argument('--eta', type=float, default=0.05)
    parser.add_argument('--tau', type=float, default=0.8)


    # directory for training
    parser.add_argument('--train_root_dir', type=str, default='dataset')
    parser.add_argument('--val_root_dir', type=str, default='dataset')
    parser.add_argument('--log_dir', type=str, default='expr/log')
    parser.add_argument('--result_dir', type=str, default='expr/results',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Nametag for the experiment')

    # step size
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--save_every_retrain', type=int, default=1000)
    parser.add_argument('--eval_every_retrain', type=int, default=100)

    args = parser.parse_args()
    args = modify_args_for_baselines(args)

    main(args)
