#!/bin/bash

seed=(1209 435 237 4840)
conflict=(0.5 1 2 5)
weight=(100 50 20 10)

for s in ${seed[@]}; do
    for ((i = 0; i < ${#conflict[@]}; ++i)); do
        conflit_pct=${conflict[$i]}
        upweight=${weight[$i]}

        echo "[Ours] seed $s, conflict_pct $conflit_pct, weight $upweight"
        echo "GPU: $1"
        echo "Data: $2"

        CUDA_VISIBLE_DEVICES=$1 python main.py --mode MRM --data $2 \
            --conflict_pct $conflit_pct \
            --seed $s --optimizer Adam \
            --lr_pre 1e-2 --pretrain_iter 10000 \
            --lr_decay_step_pre 10000 --lr_gamma_pre 0.1 --lr_main 1e-2 \
            --retrain_iter 10000 --lr_decay_step_main 10000 \
            --pseudo_label_method wrong \
            --log_dir expr/log_new --checkpoint_dir expr/checkpoints_new \
            --eval_every_retrain 500 --save_every_retrain 10000
    done
done
