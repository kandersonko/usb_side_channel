#!/usr/bin/env sh

# bad C1,D1
classifier="encoder"

dataset="dataset_d1"
method="encoder"

if [[ $dataset == "dataset_a" ]]; then
    target_label="category"
else
    target_label="device"
fi

# base_model="encoder"
# base_model="lstm"
# base_model="cnn_lstm"
# base_model="lstm_cnn"
base_model="parallel_cnn_lstm"

max_epochs=20
min_epochs=5

learning_rate=0.001

accumulate_grad_batches=2

batch_size=16
conv1_out_channels=128
conv2_out_channels=256
bottleneck_dim=128

num_lstm_layers=1
dropout=0.05

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
