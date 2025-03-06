#!/bin/bash



for b in gcn
do
    for c in german
    do

        for flag_x in 1 2 3 4
        do
            d="${b}_${c}_${flag_x}"

            echo "start ${d}"
            output_file="exp_logs_param_study/experiments_${d}.txt"
            most_idle_gpu=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{print $1 "," ($2 - $3)}' | sort -t ',' -k2,2rn | head -n 1 | awk -F',' '{print $1}')
            CUDA_VISIBLE_DEVICES=$most_idle_gpu nohup python gnn_certification_param_study.py --seed 1 --gnn $b --dataset $c --px_flag $flag_x > $output_file 2>&1 &
        
            pid=$!
            wait $pid

        done
    done
done


for b in gcn
do
    for c in credit
    do

        for flag_a in 1 2 3 4
        do
            d="${b}_${c}_${flag_a}"

            echo "start ${d}"
            output_file="exp_logs_param_study/experiments_${d}.txt"
            most_idle_gpu=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits | awk -F',' '{print $1 "," ($2 - $3)}' | sort -t ',' -k2,2rn | head -n 1 | awk -F',' '{print $1}')
            CUDA_VISIBLE_DEVICES=$most_idle_gpu nohup python gnn_certification_param_study.py --seed 100 --gnn $b --dataset $c --pa_flag $flag_a > $output_file 2>&1 &
        
            pid=$!
            wait $pid

        done
    done
done

python exp_results_param_study.py