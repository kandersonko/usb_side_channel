#!/usr/bin/env sh

# good A, B, D1, D2
# bad C1
classifier="lstm"

dataset="dataset_b"
method="raw"
target_label="device"

max_epochs=100
min_epochs=20

batch_size=12
bottleneck_dim=64
conv1_out_channels=32
conv2_out_channels=64
dropout=0.05

learning_rate=0.01

# lstm
model_name='lstm-encoder'
# lstm_input_dim=195
lstm_hidden_dim=128
lstm_num_layers=4
num_lstm_layers=$lstm_num_layers
lstm_dropout=0.01




# python classifier.py --tuning='wandb' --task=identification --method=$method --dataset=$dataset --target_label=$target_label --classifier=$classifier --batch_size=$batch_size --max_epochs=$epochs --learning_rate=0.001 --bottleneck_dim=64 --conv1_out_channels=32 --conv2_out_channels=64

# python classifier.py --tuning --task=identification --method=$method --dataset=$dataset --target_label=$target_label --classifier=$classifier --batch_size=$batch_size --max_epochs=$max_epochs --min_epochs=$min_epochs --learning_rate=$learning_rate --bottleneck_dim=$bottleneck_dim --conv1_out_channels=$conv1_out_channels --conv2_out_channels=$conv2_out_channels --dropout=$dropout --lstm_input_dim=$lstm_input_dim --num_lstm_layers=$num_lstm_layers --lstm_num_layers=$lstm_num_layers --lstm_dropout=$lstm_dropout --reconstruction_loss_weight=$reconstruction_loss_weight

python classifier.py --tuning --model_name=$model_name --task=identification --method=$method --dataset=$dataset --target_label=$target_label --classifier=$classifier --batch_size=$batch_size --max_epochs=$max_epochs --min_epochs=$min_epochs --learning_rate=$learning_rate --bottleneck_dim=$bottleneck_dim --conv1_out_channels=$conv1_out_channels --conv2_out_channels=$conv2_out_channels --dropout=$dropout --lstm_hidden_dim=$lstm_hidden_dim --num_lstm_layers=$num_lstm_layers --lstm_num_layers=$lstm_num_layers --lstm_dropout=$lstm_dropout
