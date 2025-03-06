#!/bin/bash


for a in sage gcn jk
do
    for b in german bail credit
    do
        c="${a}_${b}"

        echo "start ${c}"
        output_file="exp_logs/experiments_${c}.txt"
        most_idle_gpu=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{print $1 "," ($2 - $3)}' | sort -t ',' -k2,2rn | head -n 1 | awk -F',' '{print $1}')
        CUDA_VISIBLE_DEVICES=$most_idle_gpu nohup python gnn_certification.py --gnn $a --dataset $b > $output_file 2>&1 &
    
        pid=$!
        wait $pid
        python exp_results.py
    done
done

