#!/bin/bash

seed=(111 222 333 444)


for s in ${seed[@]}; do
    echo "[Ours] seed $s"
    echo "GPU: $1"
    echo "Data: $2"

    CUDA_VISIBLE_DEVICES=$1 python main.py --mode MRM --data $2 \
        --conflict_pct 0.5 --lambda_upweight 1 \
        --pseudo_label_method wrong --seed $s --batch_size 64 \
        --lambda_sparse 1e-9 --lr_pre 1e-3 --lr_decay_step_pre 10000 --lr_gamma_pre 0.5 \
        --pretrain_iter 10000 --lr_main 1e-3 --lr_decay_step_main 10000 --retrain_iter 20000 --lr_gamma_main 0.5 \
        --imagenet --log_dir expr/log_new --checkpoint_dir expr/checkpoints_new \
        --eval_every_retrain 500 --save_every_retrain 10000
done
