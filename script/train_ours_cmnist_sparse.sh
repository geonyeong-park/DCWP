#!/bin/bash

seed=(1004 1005 1006 1007)
sparse=(5e-8 1e-7 5e-9 1e-9 1e-8)

for s in ${seed[@]}; do
    for ((i = 0; i < ${#sparse[@]}; ++i)); do
        sp=${sparse[$i]}

        CUDA_VISIBLE_DEVICES=$1 python main.py --mode prune --data cmnist \
            --conflict_pct 0.5 --lambda_upweight 0 \
            --seed $s --optimizer SGD \
            --select_with_GCE --pseudo_label_method wrong \
            --pruning_iter 0 --retrain_iter 0 \
            --log_dir expr/log_new --checkpoint_dir expr/checkpoints_new --exp_name "sp$sp"

        CUDA_VISIBLE_DEVICES=$1 python main.py --mode prune --data cmnist \
            --conflict_pct 0.5 --lambda_upweight 50 \
            --seed $s --optimizer Adam \
            --lr_pre 1e-2 --pretrain_iter 10000 \
            --lr_decay_step_pre 10000 --lr_gamma_pre 0.1 --lr_main 1e-3 \
            --retrain_iter 1000 --lr_decay_step_main 1000 \
            --pseudo_label_method wrong --lambda_sparse $sp \
            --log_dir expr/log_new --checkpoint_dir expr/checkpoints_new --exp_name "sp$sp"
    done
done
