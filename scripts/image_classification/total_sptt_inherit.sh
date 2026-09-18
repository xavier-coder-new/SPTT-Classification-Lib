#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model="SpttLSTM"
data_names=("cifar10" "sequential_mnist")
seeds=(2023 2024 2025)
epochs=300
patience=30
# 1 represents no truncation, and the model will process the whole sequence at once.
truncate_num=1
batch_size=256
learning_rate=0.001 
num_layers=1
krank=1
slide_window_nums=1

for data_name in "${data_names[@]}"; do
    for seed in "${seeds[@]}"; do

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

        ./scripts/image_classification/demo_sptt_inherit.sh "${args[@]}"

        echo "--------------------------------------------------------------"
        echo "Finished job: ${job_name}"
        echo "--------------------------------------------------------------"

        sleep 30
    done
done




