#!/bin/bash

seed=(1111 2222 3333 4444)
conflict=(0.5 1 2 5)

for s in ${seed[@]}; do
    for ((i = 0; i < ${#conflict[@]}; ++i)); do
        conflit_pct=${conflict[$i]}

        echo "[JTT] seed $s, conflict_pct $conflit_pct"
        echo "GPU: $1"

        CUDA_VISIBLE_DEVICES=$1 python main.py --mode JTT \
            --conflict_pct $conflit_pct --seed $s
    done
done
