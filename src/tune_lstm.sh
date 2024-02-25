#!/usr/bin/env sh

# good A, B, D1, D2
# bad C1
classifier="encoder"

dataset="dataset_a"
method="encoder"
target_label="category"

# base_model="lstm"
# base_model="cnn_lstm"
base_model="lstm_cnn"
# base_model="parallel_cnn_lstm"

max_epochs=25
min_epochs=20

learning_rate=0.01

accumulate_grad_batches=2

batch_size=16
conv1_out_channels=128
conv2_out_channels=64

bottleneck_dim=64
num_lstm_layers=1
dropout=0.25

# lstm
model_name='lstm-encoder'

lstm_hidden_dim=$bottleneck_dim
lstm_num_layers=$num_lstm_layers
lstm_dropout=$dropout



# add --log for logging to wandb

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
    --accumulate_grad_batches=$accumulate_grad_batches \
    --lstm_dropout=$lstm_dropout \
    --base_model=$base_model
