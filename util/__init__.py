import os
from os.path import join as ospj
import json
from util.params import config

def setup(args):
    fname = args.exp_name
    if fname is None:
        fname = f'{args.data}_conflict_{args.conflict_pct}_{args.mode}_seed_{args.seed}'
    else:
        fname = f'{args.data}_conflict_{args.conflict_pct}_{args.mode}_{fname}_seed_{args.seed}'

    args.result_dir = ospj(args.result_dir, fname)
    os.makedirs(args.result_dir, exist_ok=True)

    args.log_dir = ospj(args.log_dir, fname)
    os.makedirs(args.log_dir, exist_ok=True)

    args.checkpoint_dir = ospj(args.checkpoint_dir, fname)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args

def save_config(args):
    with open(os.path.join(args.log_dir, 'args.txt'), 'a') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(args.log_dir, 'param.txt'), 'a') as f:
        json.dump(config, f, indent=2)

def modify_args_for_baselines(args):
    if args.mode == 'featureswap':
        args.lambda_swap = 1
        args.lambda_dis_align = 1
        args.lambda_swap_align = 1

        args.total_iters = 50000
        args.swap_iter = 10000
        args.lr = 1e-3
        args.lr_decay_step = 10000
        args.lr_gamma = 0.5

    return args

