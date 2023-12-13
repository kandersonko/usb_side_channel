#!/usr/bin/env sh

# list of classifiers
classifiers=("random_forest" "SVC" "KNN" "gradient_boosting" "decision_tree")

# list of datasets
datasets=("dataset_a-category" "dataset_b-device" "dataset_c1-device" "dataset_c2-device" "dataset_d1-device" "dataset_d2-device")

for classifier in "${classifiers[@]}"; do
    for dataset in "${datasets[@]}"; do
        # if "category" is in name of dataset, then target_label is "category", else it is "device"
        if [[ $dataset == *"category"* ]]; then
            target_label="category"
        else
            target_label="device"
        fi
        echo "=============================================="
        echo "=============================================="
        echo "Running classifier $classifier on dataset $dataset with target label $target_label"
        python classifier.py --task=identification --method=raw --dataset=$dataset --target_label=$target_label --classifier=$classifier

    done
done
