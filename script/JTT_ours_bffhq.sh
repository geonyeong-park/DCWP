#!/bin/bash

seed=(111 222 333 444)
stop_iter=(500 1000 1500)

for s in ${seed[@]}; do
    for it in ${stop_iter[@]}; do
        echo "[Ours] seed $s, stop_iter $it"
        echo "GPU: $1"
        echo "Data: $2"

        CUDA_VISIBLE_DEVICES=$1 python main.py --mode prune --data $2 \
            --conflict_pct 0.5 --lambda_upweight 80 \
            --pseudo_label_method wrong --select_with_GCE --seed $s --batch_size 64 \
            --lambda_sparse 1e-9 --lr_pre 1e-3 --lr_decay_step_pre 10000 --lr_gamma_pre 0.5 \
            --pretrain_iter 10000 --lr_main 5e-4 --lr_decay_step_main 10000 --retrain_iter 1000 \
            --imagenet --log_dir expr/log_new --checkpoint_dir expr/checkpoints_new \
            --earlystop_iter $it --exp_name "GCE_stop_$it"
    done
done
