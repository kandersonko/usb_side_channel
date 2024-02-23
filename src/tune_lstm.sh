#!/usr/bin/env sh

# good A, B, D1, D2
# bad C1
classifier="lstm"

dataset="dataset_d2"
method="encoder"
target_label="device"

max_epochs=200
min_epochs=20

learning_rate=0.01

batch_size=32
conv1_out_channels=32
conv2_out_channels=16

bottleneck_dim=64
num_lstm_layers=2
dropout=0.2

# lstm
model_name='lstm-encoder'
# lstm_input_dim=195
lstm_hidden_dim=$bottleneck_dim
lstm_num_layers=$num_lstm_layers
lstm_dropout=$dropout


python classifier.py \
    --tuning \
    --model_name=$model_name \
    --task=identification \
    --method=$method \
    --dataset=$dataset \
    --target_label=$target_label \
    --classifier=$classifier \
    --batch_size=$batch_size \
    --max_epochs=$max_epochs \
    --min_epochs=$min_epochs \
    --learning_rate=$learning_rate \
    --bottleneck_dim=$bottleneck_dim \
    --conv1_out_channels=$conv1_out_channels \
    --conv2_out_channels=$conv2_out_channels \
    --dropout=$dropout \
    --lstm_hidden_dim=$lstm_hidden_dim \
    --num_lstm_layers=$num_lstm_layers \
    --lstm_num_layers=$lstm_num_layers \
    --lstm_dropout=$lstm_dropout \
    --use_encoder
