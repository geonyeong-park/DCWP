#!/bin/bash

seed=(1111 2222 3333 4444)
conflict=(0.5 1 2 5)

for s in ${seed[@]}; do
    for ((i = 0; i < ${#conflict[@]}; ++i)); do
        conflit_pct=${conflict[$i]}

        echo "[FeatureSwap] seed $s, conflict_pct $conflit_pct"
        echo "GPU: $1"
        echo "Data: $2"

        CUDA_VISIBLE_DEVICES=$1 python main.py --mode featureswap --data $2 \
            --conflict_pct $conflit_pct --seed $s \
            --lambda_dis_align 1 --lambda_swap_align 1 \
            --exp_name lambda1
    done
done
