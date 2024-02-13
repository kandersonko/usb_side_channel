#!/usr/bin/env sh

classifier="lstm"
dataset="dataset_b"
target_label="device"
method="raw"
epochs=10


python classifier.py --tuning='wandb' --task=identification --method=$method --dataset=$dataset --target_label=$target_label --classifier=$classifier --batch_size=8 --max_epochs=$epochs
