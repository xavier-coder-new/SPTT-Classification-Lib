#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

models=("SpttLSTM")
data_names=("cifar10")
seeds=(2023)
# epochs=500
# patience=50
# 1 represents no truncation, and the model will process the whole sequence at once.
truncate_num=1
batch_size=256
# batch_size=512
# learning_rate=0.005
learning_rate=0.001
num_layers=3
krank=1
slide_window_nums=1

args=(
    "$model" "$data_name" "$seed" "$epochs"
    "$patience" "$truncate_num" "$batch_size"
    "$learning_rate" "$num_layers" "$krank"
    "$slide_window_nums"
)

for data_name in "${data_names[@]}"; do
  for seed in "${seeds[@]}"; do
    for model in "${models[@]}"; do
        if [[ "$model" = "SpttLSTM" ]] || [[ "$model" = "SpttGRU" ]]; then
            learning_rate=0.001
            epochs=500
            patience=50
        elif [[ "$model" = "BpttLSTM" ]] || [[ "$model" = "BpttGRU" ]]; then
            learning_rate=0.001
            epochs=500
            patience=50
        fi

        job_name="image_classification---${model}---${data_name}---${seed}"
        echo "--------------------------------------------------------------"
        echo "Running job: ${job_name}"
        echo "Truncate Num: ${truncate_num}"
        echo "Learning rate: ${learning_rate}"
        echo "Num layers: ${num_layers}"
        echo "Krank: ${krank}"
        echo "Slide window nums: ${slide_window_nums}"
        echo "--------------------------------------------------------------"

        args=(
            "$model" "$data_name" "$seed" "$epochs"
            "$patience" "$truncate_num" "$batch_size"
            "$learning_rate" "$num_layers" "$krank"
            "$slide_window_nums"
        )

        ./scripts/image_classification/test.sh "${args[@]}"

        echo "--------------------------------------------------------------"
        echo "Finished job: ${job_name}"
        echo "--------------------------------------------------------------"

        sleep 30

        used_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1)
        if [[ "$used_mem" =~ ^[0-9]+$ ]] && [ "$used_mem" -gt 1000 ]; then
            echo "GPU Memory still high ($used_mem MB), waiting extra 30s..."
            sleep 30
        fi

    done
  done
done




