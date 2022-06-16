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
            --seed $s --optimizer SGD
    done
done
