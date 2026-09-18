#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model=$1
data_name=$2
seed=$3
epochs=$4
patience=$5
# for end mode, the truncate_num need to set to 1.
truncate_num=$6
batch_size=$7
learning_rate=$8
# hidden_dim=512
hidden_dim=256
embed_dim=256
num_layers=$9
krank="${10}"
# pixel, row
seq_mode=pixel
vali_ratio=0.1
# for end mode, the recommended number of sliding windows is 1
slide_window_nums="${11}"
exp_type=image
inherit_bptt=1.0

# Set dataset-specific parameters
if [[ "$data_name" = "sequential_mnist" ]] || [[ "$data_name" = "cifar10" ]]; then
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
        --embed_dim $embed_dim \
        --krank $krank \
        --truncate_num $truncate_num \
        --seed $seed \
        --patience $patience \
        --vali_ratio $vali_ratio \
        --seq_mode $seq_mode \
        --feature_dim $feature_dim \
        --output_dim $output_dim \
        --num_layers $num_layers \
        --use_rich \
        --permute \
        --slide_window_nums $slide_window_nums \
        --exp_type=$exp_type \
        --need_vali \
        --to_grayscale \
        --inherit_bptt $inherit_bptt \
        --bptt_low_rank
else
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
        --vali_ratio $vali_ratio \
        --seq_mode $seq_mode \
        --feature_dim $feature_dim \
        --output_dim $output_dim \
        --num_layers $num_layers \
        --use_rich \
        --slide_window_nums $slide_window_nums \
        --exp_type=$exp_type \
        --permute \
        --to_grayscale \
        --inherit_bptt $inherit_bptt \
        --bptt_low_rank
fi
