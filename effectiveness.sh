#!/bin/bash

set -e

mkdir -p exp_logs

for a in gcn jk sage
do
    for b in german bail credit
    do
        c="${a}_${b}"
        echo "=============================="
        echo "Start experiment: ${c}"

        output_file="exp_logs/experiments_${c}.txt"

        most_idle_gpu=$(nvidia-smi --query-gpu=uuid,power.draw --format=csv,noheader,nounits | sort -t',' -k2,2n | head -n1 | cut -d',' -f1)
        echo "Using physical GPU UUID ${most_idle_gpu}"

        CUDA_VISIBLE_DEVICES=$most_idle_gpu \
        python gnn_certification.py --gnn "$a" --dataset "$b" \
            > "$output_file" 2>&1

        echo "Experiment ${c} finished, running exp_results.py"
        python exp_results.py
    done
done

echo "All experiments finished."
