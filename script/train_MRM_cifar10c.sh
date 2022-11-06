#!/bin/bash

seed=(555 666 777 888)
conflict=(0.5 1 2 5)
weight=(80 50 30 10)


for s in ${seed[@]}; do
    for ((i = 0; i < ${#conflict[@]}; ++i)); do
        conflict_pct=${conflict[$i]}
        upweight=${weight[$i]}

        echo "[Ours] seed $s, conflict_pct $conflict_pct, weight $upweight"
        echo "GPU: $1"
        echo "Data: $2"

        CUDA_VISIBLE_DEVICES=$1 python main.py --mode MRM --data $2 \
            --conflict_pct $conflict_pct --lambda_upweight 1 \
            --pseudo_label_method wrong --seed $s \
            --lambda_sparse 1e-9 --lr_pre 1e-3 --lr_decay_step_pre 10000 --lr_gamma_pre 0.5 \
            --pretrain_iter 10000 --lr_main 1e-3 --lr_decay_step_main 10000 --lr_gamma_main 0.5 --retrain_iter 20000 \
            --imagenet --log_dir expr/log_new --checkpoint_dir expr/checkpoints_new \
            --eval_every_retrain 500 --save_every_retrain 10000
    done
done
