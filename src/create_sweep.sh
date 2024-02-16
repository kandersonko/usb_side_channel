#!/usr/bin/env sh

name=$1
sweep_file=$2

project="usb_side_channel"

if [ -z "$name" ]; then
    echo "Please provide a name for the sweep"
    exit 1
fi

if [ -z "$sweep_file" ]; then
    echo "Please provide a sweep file"
    exit 1
fi

# append timestamp to name
name="${name}--$(date +'%Y-%m-%d_%H-%M-%S')"

# Initialize sweep and get the sweep id
wandb sweep --project $project $sweep_file --name $name
