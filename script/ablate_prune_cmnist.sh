#!/bin/bash

iters=(500 1000 1500 2000)

for it in ${iters[@]}; do
    echo "[Ablation] pruning iteration: $it"
    echo "GPU: $1"
    echo "Data: $2"

    CUDA_VISIBLE_DEVICES=$1 python main.py --mode prune --data $2 \
        --conflict_pct 1 --lambda_upweight 80 --optimizer SGD \
        --pruning_iter $it --exp_name layerwise
done
