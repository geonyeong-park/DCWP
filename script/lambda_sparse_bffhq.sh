#!/bin/bash
ld=(1. 0.5 0.1 0.05 0.01)

for l in ${ld[@]}; do
    echo "[Ours] Lambda_sparse $l, conflict_pct 0.5"
    echo "GPU: $1"
    echo "Data: $2"
    CUDA_VISIBLE_DEVICES=$1 python main.py --mode prune --data $2 \
        --conflict_pct 0.5 --lambda_upweight 80 \
        --select_with_GCE --pseudo_label_method wrong --seed 2022 --batch_size 64 \
        --lambda_sparse $l --lr_pre 1e-3 --lr_decay_step_pre 10000 --lr_gamma_pre 0.5 \
        --pretrain_iter 10000 --lr_main 5e-4 --lr_decay_step_main 10000 --retrain_iter 1000 \
        --imagenet --exp_name "sparse_$l" --log_dir expr/log_new
done
