#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model="BpttLSTM"
data_names=("sequential_mnist")
seeds=(2023)
epochs=300
patience=30
# 1 represents no truncation, and the model will process the whole sequence at once.
truncate_num=1
batch_size=256
learning_rate=0.001
num_layers=1
krank=1
slide_window_nums=1
# 0.1 0.5 1.0 5.0
clip_norms=(0.1)

for seed in "${seeds[@]}"; do
  for data_name in "${data_names[@]}"; do
    for clip_norm in "${clip_norms[@]}"; do
        job_name="image_classification---${model}---${data_name}---${seed}"
        echo "--------------------------------------------------------------"
        echo "Running job: ${job_name}"
        echo "Truncate Num: ${truncate_num}"
        echo "Learning rate: ${learning_rate}"
        echo "Num layers: ${num_layers}"
        echo "Krank: ${krank}"
        echo "Slide window nums: ${slide_window_nums}"
        echo "Gradient clipping max norm: ${clip_norm}"
        echo "--------------------------------------------------------------"

        args=(
            "$model" "$data_name" "$seed" "$epochs"
            "$patience" "$truncate_num" "$batch_size"
            "$learning_rate" "$num_layers" "$krank"
            "$slide_window_nums" "$clip_norm"
        )

        ./scripts/image_classification/demo_gradient.sh "${args[@]}"

        echo "--------------------------------------------------------------"
        echo "Finished job: ${job_name}"
        echo "--------------------------------------------------------------"

        sleep 30

        # used_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1)
        # if [[ "$used_mem" =~ ^[0-9]+$ ]] && [ "$used_mem" -gt 1000 ]; then
        #     echo "GPU Memory still high ($used_mem MB), waiting extra 30s..."
        #     sleep 30
        # fi
    done
  done
done




