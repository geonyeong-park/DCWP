#!/bin/bash
ld=(1. 0.5 0.1 0.05 0.01)
conflict=(0.5 1 2 5)
weight=(80 50 30 10)
for l in ${ld[@]}; do
    for ((i = 0; i < ${#conflict[@]}; ++i)); do
        conflit_pct=${conflict[$i]}
        upweight=${weight[$i]}
        echo "[Ours] Lambda_sparse $l, conflict_pct $conflit_pct, weight $upweight"
        echo "GPU: $1"
        echo "Data: $2"
        CUDA_VISIBLE_DEVICES=$1 python main.py --mode prune --data $2 \
            --conflict_pct $conflit_pct --lambda_upweight $upweight \
            --pseudo_label_method ensemble --seed 2022 \
            --lambda_sparse $l --lr_pre 1e-3 --lr_decay_step_pre 10000 --lr_gamma_pre 0.5 \
            --pretrain_iter 10000 --lr_main 5e-4 --lr_decay_step_main 10000 --retrain_iter 1000 \
            --imagenet --exp_name "sparse_$l" --log_dir expr/log_new
    done
done
