#!/bin/bash

seed=(55 66 77 88)
conflict=(0.5 1 2 5)
weight=(80 50 30 10)


for s in ${seed[@]}; do
    for ((i = 0; i < ${#conflict[@]}; ++i)); do
        conflict_pct=${conflict[$i]}
        upweight=${weight[$i]}

        CUDA_VISIBLE_DEVICES=$1 python main.py --mode prune --data cmnist \
            --conflict_pct $conflict_pct --lambda_upweight 0 \
            --seed $s --optimizer SGD \
            --select_with_GCE --pseudo_label_method wrong \
            --pruning_iter 0 --retrain_iter 0 \
            --log_dir expr/log_new --checkpoint_dir expr/checkpoints_new

        CUDA_VISIBLE_DEVICES=$1 python main.py --mode prune --data cmnist \
            --conflict_pct $conflict_pct --lambda_upweight $upweight \
            --seed $s --optimizer Adam \
            --lr_pre 1e-2 --pretrain_iter 10000 \
            --lr_decay_step_pre 10000 --lr_gamma_pre 0.1 --lr_main 1e-3 \
            --retrain_iter 1000 --lr_decay_step_main 1000 \
            --pseudo_label_method wrong \
            --log_dir expr/log_new --checkpoint_dir expr/checkpoints_new
    done
done
