#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model=BpttLSTM
data_name=google_speech
seed=2025
epochs=4
patience=2
# for end mode, the truncate_num need to set to 1.
truncate_num=1
batch_size=256
learning_rate=0.001
hidden_dim=512
embed_dim=256
num_layers=1
krank=7
# for end mode, the recommended number of sliding windows is 1
slide_window_nums=1
exp_type=audio

# Set dataset-specific parameters
if [[ "$data_name" = "google_speech" ]]; then
    feature_dim=64
    output_dim=36
fi

# Run training with or without validation set
python -m run \
    --model $model \
    --data_name $data_name \
    --seed $seed \
    --batch_size $batch_size \
    --epochs $epochs \
    --lr_rate $learning_rate \
    --hidden_dim $hidden_dim \
    --embed_dim $embed_dim \
    --krank $krank \
    --truncate_num $truncate_num \
    --seed $seed \
    --patience $patience \
    --feature_dim $feature_dim \
    --output_dim $output_dim \
    --num_layers $num_layers \
    --use_rich \
    --slide_window_nums $slide_window_nums \
    --exp_type=$exp_type \
