#!/usr/bin/env sh

# list of classifiers
# classifiers=("random_forest" "SVC" "KNN" "gradient_boosting" "decision_tree" "lstm")
classifiers=("lstm")

# list of datasets
# datasets=("dataset_a" "dataset_b" "dataset_c1" "dataset_c2" "dataset_d1" "dataset_d2")
datasets=("dataset_a")

# methods=("raw" "encoder" "tsfresh")
methods=("tsfresh")

for method in "${methods[@]}"; do
    for classifier in "${classifiers[@]}"; do
        for dataset in "${datasets[@]}"; do
            # if "category" is in name of dataset, then target_label is "category", else it is "device"
            if [[ $dataset == "dataset_a" ]]; then
                target_label="category"
            else
                target_label="device"
            fi

            echo "=============================================="
            echo "=============================================="
            echo "Running classifier $classifier on dataset $dataset with target label $target_label using method $method"
            python classifier.py --task=identification --method=$method --dataset=$dataset --target_label=$target_label --classifier=$classifier --batch_size=8
        done
    done
done
