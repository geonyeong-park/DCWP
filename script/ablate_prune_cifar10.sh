#!/bin/bash

iters=(2000 500 1000 1500)

for it in ${iters[@]}; do
    echo "[Ablation] pruning iteration: $it"
    echo "GPU: $1"
    echo "Data: $2"

    CUDA_VISIBLE_DEVICES=$1 python main.py --mode prune --data $2 \
        --conflict_pct 1 --lambda_upweight 50 --optimizer SGD \
        --pruning_iter $it --exp_name layerwise \
        --lambda_sparse 1e-9 --lr_pre 1e-3 --lr_decay_step_pre 10000 --lr_gamma_pre 0.5 \
        --pretrain_iter 10000 --lr_main 5e-4 --lr_decay_step_main 10000 --retrain_iter 1000 \
        --imagenet --seed 5555
done
