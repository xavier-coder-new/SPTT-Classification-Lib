#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model=$1
data_name=$2
seed=$3
epochs=$4
patience=$5
truncate_num=$6
batch_size=$7
learning_rate=$8
hidden_dim=512
embed_dim=256
num_layers=$9
krank="${10}"
vali_ratio=0.1
exp_type=text
slide_window_nums="${11}"
# pad, repeat
length_mode=pad

if [[ "$data_name" = "imdb" ]]; then
    batch_size=128
    feature_dim=$embed_dim
    output_dim=2
    need_vali=True
elif [[ "$data_name" = "ag_news" ]]; then
    batch_size=256
    feature_dim=$embed_dim
    output_dim=4
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
        --feature_dim $feature_dim \
        --output_dim $output_dim \
        --num_layers $num_layers \
        --use_rich \
        --permute \
        --slide_window_nums $slide_window_nums \
        --exp_type $exp_type \
        --length_mode $length_mode \
        --need_vali \
        --offline
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
        --feature_dim $feature_dim \
        --output_dim $output_dim \
        --num_layers $num_layers \
        --use_rich \
        --permute \
        --slide_window_nums $slide_window_nums \
        --exp_type $exp_type \
        --length_mode $length_mode \
        --offline
fi
