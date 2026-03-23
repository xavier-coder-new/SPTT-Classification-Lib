#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

models=("BpttLSTM" "SpttLSTM" "SpttLSTM_End" "SpttGRU" "BpttGRU" "SpttGRU_End")
data_names=("google_speech")
seeds=(2023 2024 2025)
epochs=300
patience=30
# 1 represents no truncation, and the model will process the whole sequence at once.
truncate_num=1
# batch_size=128
batch_size=256
# learning_rate=0.005
# for 0.01, the SpttLSTM encountered the error, which One of values of Sigma_ih is nan in 5090 server.
learning_rate=0.001
num_layers=1
krank=7
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
        job_name="audio_classification---${model}---${data_name}---${seed}"
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

        ./scripts/audio_classification/test.sh "${args[@]}"

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




