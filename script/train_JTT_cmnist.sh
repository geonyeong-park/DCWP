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

        #CUDA_VISIBLE_DEVICES=$1 python main.py --mode JTT --data $2 \
        #    --conflict_pct $conflit_pct --lambda_upweight 0 \
        #    --seed $s --optimizer SGD \
        #    --pseudo_label_method wrong --pretrain_iter 2000 \
        #    --pruning_iter 0 --retrain_iter 0 \
        #    --log_dir expr/log_new --checkpoint_dir expr/checkpoints_new

        CUDA_VISIBLE_DEVICES=$1 python main.py --mode JTT --data $2 \
            --conflict_pct $conflit_pct --lambda_upweight $upweight \
            --seed $s --optimizer Adam \
            --lr_pre 1e-2 --pretrain_iter 2000 \
            --lr_decay_step_pre 10000 --lr_gamma_pre 0.1 --lr_main 1e-3 \
            --retrain_iter 10000 --lr_decay_step_main 10000 --lr_gamma_main 0.1 \
            --pseudo_label_method wrong --eval_every_retrain 500 --save_every_retrain 500 \
            --weight_decay 1.
            --log_dir expr/log_new --checkpoint_dir expr/checkpoints_new
    done
done
