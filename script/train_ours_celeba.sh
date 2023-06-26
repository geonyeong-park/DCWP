#!/bin/bash

seed=(1006 1007 1008 1009)


for s in ${seed[@]}; do
    echo "[Ours] seed $s"
    echo "GPU: $1"
    echo "Data: $2"

    #TODO: Now: lr_pre=1e-4. How about 1e-3?
    CUDA_VISIBLE_DEVICES=$1 python main.py --mode prune --data $2 \
        --conflict_pct 1 --lambda_upweight 20 \
        --pseudo_label_method wrong --select_with_GCE --seed $s --batch_size 128 \
        --lambda_sparse 1e-9 --lr_pre 1e-4 --lr_decay_step_pre 10000 --lr_gamma_pre 0.5 \
        --pretrain_iter 10000 --lr_main 5e-4 --lr_decay_step_main 10000 --retrain_iter 1000 \
        --imagenet --log_dir expr/log_new --checkpoint_dir expr/checkpoints_new
done
