#!/usr/bin/env sh

# good A, B, D1, D2
# bad C1
classifier="lstm"
dataset="dataset_c1"
target_label="device"
method="raw"
epochs=100
batch_size=8

python classifier.py --tuning='wandb' --task=identification --method=$method --dataset=$dataset --target_label=$target_label --classifier=$classifier --batch_size=$batch_size --max_epochs=$epochs --learning_rate=0.001 --bottleneck_dim=64 --conv1_out_channels=32 --conv2_out_channels=64
# python classifier.py --task=identification --method=$method --dataset=$dataset --target_label=$target_label --classifier=$classifier --batch_size=8 --max_epochs=$epochs
