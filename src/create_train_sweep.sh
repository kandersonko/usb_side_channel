#!/usr/bin/env sh

name="usb_side_channel_train_sweep-$(date +'%Y-%m-%d_%H-%M-%S')"

# Initialize sweep and get the sweep id
wandb sweep --project usb_side_channel train_sweep.yaml --name $name
