#!/bin/bash


seed=(29 39 49)
#ld=(1e-9 3e-9 5e-9 1e-8 3e-8 5e-8 1e-7)
ld=(5e-6)


for s in ${seed[@]}; do
    for ((i = 0; i < ${#ld[@]}; ++i)); do
        ld_sp=${ld[$i]}

        CUDA_VISIBLE_DEVICES=$1 python main.py --mode prune --data $2 \
            --conflict_pct 1 --lambda_upweight 50 \
            --pseudo_label_method ensemble --seed $s \
            --lambda_sparse 1e-9 --lr_pre 1e-3 --lr_decay_step_pre 10000 --lr_gamma_pre 0.5 \
            --pretrain_iter 10000 --lr_main 5e-4 --lr_decay_step_main 10000 --retrain_iter 1000 \
            --imagenet --log_dir expr/log_new --checkpoint_dir expr/checkpoints_new \
            --exp_name "ld_$ld_sp" --lambda_sparse $ld_sp
    done
done
