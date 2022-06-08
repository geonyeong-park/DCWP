#!/bin/bash

seed=(1111 2222 3333 4444)
conflict=(0.5 1 2 5)
weight=(80 50 30 10)


for s in ${seed[@]}; do
    for ((i = 0; i < ${#conflict[@]}; ++i)); do
        conflit_pct=${conflict[$i]}
        upweight=${weight[$i]}

        echo "[Ours] seed $s, conflict_pct $conflit_pct, weight $upweight"
        echo "GPU: $1"
        echo "Data: $2"

        CUDA_VISIBLE_DEVICES=$1 python main.py --mode prune --data $2 \
            --conflict_pct $conflit_pct --lambda_upweight $upweight \
            --select_with_GCE --seed $s \
            --lambda_sparse 1e-9 --lr_pre 1e-3 --lr_decay_step_pre 10000 --lr_gamma_pre 0.5 \
            --pretrain_iter 10000 --lr_main 5e-4 --lr_decay_step_main 10000 --retrain_iter 5000 \
            --imagenet
    done
done
