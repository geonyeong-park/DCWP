#!/bin/bash

seed=(11 22 33 44)
ld=(1e-7 1e-8 1e-9 5e-9 7e-9 5e-8 5e-7)


for s in ${seed[@]}; do
    for ((i = 0; i < ${#ld[@]}; ++i)); do
        ld_sp=${ld[$i]}
        echo "[Ours] seed $s"
        echo "GPU: $1"
        echo "Data: $2"

        CUDA_VISIBLE_DEVICES=$1 python main.py --mode prune --data $2 \
            --conflict_pct 0.5 --lambda_upweight 80 \
            --pseudo_label_method wrong --select_with_GCE --seed $s --batch_size 128 \
            --lambda_sparse 1e-9 --lr_pre 1e-3 --lr_decay_step_pre 10000 --lr_gamma_pre 0.5 \
            --pretrain_iter 10000 --lr_main 5e-4 --lr_decay_step_main 10000 --retrain_iter 1000 \
            --imagenet --log_dir expr/log_new --checkpoint_dir expr/checkpoints_new \
            --exp_name "ld_$ld_sp" --lambda_sparse $ld_sp \
            --train_root_dir /home/pky/research_new/dataset --val_root_dir /home/pky/research_new/dataset
        done
done
