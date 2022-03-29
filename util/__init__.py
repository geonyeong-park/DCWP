import os
from os.path import join as ospj
import json
from util.params import config

def setup(args):
    fname = args.exp_name
    if fname is None:
        fname = f'{args.data}_LabeledRatio_{args.labeled_ratio}_conflict_{args.conflict_pct}_seed_{args.seed}'

    args.result_dir = ospj(args.result_dir, fname)
    os.makedirs(args.result_dir, exist_ok=True)

    args.log_dir = ospj(args.log_dir, fname)
    os.makedirs(args.log_dir, exist_ok=True)

    args.checkpoint_dir = ospj(args.checkpoint_dir, fname)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.labeled_ratio == 1.:
        args.use_unsup_data = False
    else:
        args.use_unsup_data = True

    return args

def save_config(args):
    with open(os.path.join(args.log_dir, 'args.txt'), 'a') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(args.log_dir, 'param.txt'), 'a') as f:
        json.dump(config, f, indent=2)

