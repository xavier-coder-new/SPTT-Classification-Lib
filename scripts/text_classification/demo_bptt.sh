#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model=BpttLSTM
data_name=ag_news
seed=2024
epochs=100
patience=10
truncate_num=1
batch_size=256
learning_rate=0.001
hidden_dim=512
embed_dim=256
num_layers=1
krank=6
vali_ratio=0.1
exp_type=text
slide_window_nums=1
length_mode=repeat

if [[ "$data_name" = "ag_news" ]]; then
    batch_size=256
    feature_dim=$embed_dim
    output_dim=4
    need_vali=True
    max_length=1200
    fixed_length=1200
elif [[ "$data_name" = "long_listops" ]]; then
    batch_size=256
    feature_dim=$embed_dim
    output_dim=10
    need_vali=True
    max_length=2000
    fixed_length=2000
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
        --offline \
        --max_length $max_length \
        --fixed_length $fixed_length \
        # --profile_bptt_compute
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
        --slide_window_nums $slide_window_nums \
        --exp_type $exp_type \
        --length_mode $length_mode \
        --permute \
        --offline \
        --max_length $max_length \
        --fixed_length $fixed_length \
        # --profile_bptt_compute 
fi
