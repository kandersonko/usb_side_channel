#!/usr/bin/env sh

# list of classifiers
# classifiers=("random_forest" "SVC" "KNN" "gradient_boosting" "decision_tree" "lstm")
classifiers=("random_forest" "SVC" "KNN" "gradient_boosting" "decision_tree")
# classifiers=("lstm")

# list of datasets
datasets=("dataset_a" "dataset_b" "dataset_c1" "dataset_c2" "dataset_d1" "dataset_d2")
# datasets=("dataset_b")

methods=("raw" "encoder" "tsfresh")
# methods=("encoder")

folds=10
epochs=100

# lstm config
learning_rate=0.001
batch_size=12
bottleneck_dim=163
conv1_out_channels=103
conv2_out_channels=196
dropout=0.05
lstm_dropout=0.05
lstm_hidden_dim=163
num_lstm_layers=5
lstm_num_layers=5
reconstruction_loss_weight=0.78

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
            # python classifier.py --task=identification --method=$method --dataset=$dataset --target_label=$target_label --classifier=$classifier --batch_size=8 --kfold=$folds --max_epochs=$epochs --learning_rate=0.001 --bottleneck_dim=64 --conv1_out_channels=32 --conv2_out_channels=64
            python classifier.py --task=identification --method=$method --dataset=$dataset --target_label=$target_label --classifier=$classifier --batch_size=$batch_size --kfold=$folds --max_epochs=$epochs --learning_rate=$learning_rate --bottleneck_dim=$bottleneck_dim --conv1_out_channels=$conv1_out_channels --conv2_out_channels=$conv2_out_channels --dropout=$dropout --lstm_dropout=$lstm_dropout --lstm_hidden_dim=$lstm_hidden_dim --num_lstm_layers=$num_lstm_layers --lstm_num_layers=$lstm_num_layers --reconstruction_loss_weight=$reconstruction_loss_weight
        done
    done
done

# create tables
# python create_tables.py -i results
