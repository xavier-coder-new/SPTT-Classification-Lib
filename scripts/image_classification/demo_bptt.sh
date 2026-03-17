#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model=BpttLSTM
data_name=sequential_mnist
seed=2025
epochs=1
patience=1
truncate_num=20
batch_size=128
learning_rate=0.001
hidden_dim=512
embedding_dim=256
max_length=784
num_layers=2
krank=7
# pixel, row
seq_mode=pixel
vali_ratio=0.1

# Set dataset-specific parameters
if [[ "$data_name" = "sequential_mnist" ]]; then
    feature_dim=1
    output_dim=10
    need_vali=True
fi

# Run training with or without validation set
if [ "$need_vali" = "True" ]; then
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
        --max_length $max_length \
        --vali_ratio $vali_ratio \
        --seq_mode $seq_mode \
        --feature_dim $feature_dim \
        --output_dim $output_dim \
        --num_layers $num_layers \
        --use_rich \
        --permute \
        --need_vali 
else
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
        --max_length $max_length \
        --vali_ratio $vali_ratio \
        --seq_mode $seq_mode \
        --feature_dim $feature_dim \
        --output_dim $output_dim \
        --num_layers $num_layers \
        --use_rich \
        --permute 
fi
