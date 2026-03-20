#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model=SpttLSTM
data_name=google_speech
seed=2025
epochs=4
patience=2
# for end mode, the truncate_num need to set to 1.
truncate_num=1
batch_size=128
learning_rate=0.001
hidden_dim=512
embedding_dim=256
max_length=784
num_layers=1
krank=7
# pixel, row
seq_mode=pixel
vali_ratio=0.1
# for end mode, the recommended number of sliding windows is 1
slide_window_nums=4

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
    --embedding_dim $embedding_dim \
    --krank $krank \
    --truncate_num $truncate_num \
    --seed $seed \
    --patience $patience \
    --feature_dim $feature_dim \
    --output_dim $output_dim \
    --num_layers $num_layers \
    --use_rich \
    --slide_window_nums $slide_window_nums \

