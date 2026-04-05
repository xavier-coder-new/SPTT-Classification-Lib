#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model=SpttLSTM
data_name=esc50
seed=2025
epochs=300
patience=30
# for end mode, the truncate_num need to set to 1.
truncate_num=1
batch_size=256
learning_rate=0.001
hidden_dim=256
embed_dim=256
num_layers=1
krank=6
# for end mode, the recommended number of sliding windows is 1
slide_window_nums=1
exp_type=audio


# Set dataset-specific parameters
if [[ "$data_name" = "google_speech" ]]; then
    batch_size=256
    feature_dim=64
    output_dim=36
    sample_rate=16000
    n_mels=64
    n_fft=512
    hop_length=160
elif [[ "$data_name" = "esc50" ]]; then
    batch_size=64
    feature_dim=128
    output_dim=50
    sample_rate=44100
    n_mels=128
    n_fft=2048
    hop_length=1024
elif [[ "$data_name" = "nsynth" ]]; then
    batch_size=64
    feature_dim=64
    output_dim=11
    sample_rate=16000
    n_mels=64
    n_fft=1024
    # hop_length smaller, the sequence length will be longer.
    # hop_length=512
    hop_length=160
else
    echo "Unsupported dataset: $data_name"
    exit 1
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
    --n_mels $n_mels \
    --sample_rate $sample_rate \
    --n_fft $n_fft \
    --hop_length $hop_length \
    --run_five_fold

