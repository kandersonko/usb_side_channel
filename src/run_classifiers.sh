#!/usr/bin/env sh

# Make sure the run the script with a specific number of CPU cores (e.g., 4)

# list of classifiers
# classifiers=("random_forest" "SVC" "KNN" "gradient_boosting" "decision_tree" "lstm")
# classifiers=("random_forest" "SVC" "KNN" "gradient_boosting" "decision_tree")
classifiers=("random_forest")
# classifiers=("lstm")

# list of datasets
datasets=("dataset_a" "dataset_b" "dataset_c1" "dataset_c2" "dataset_d1" "dataset_d2")
# datasets=("dataset_d1" "dataset_d2")
# datasets=("dataset_c2" "dataset_d1")

methods=("raw" "encoder" "tsfresh")
# methods=("tsfresh")

folds=10

base_model="parallel_cnn_lstm"

max_epochs=50
min_epochs=5

early_stopping_patience=5
learning_rate_patience=3

learning_rate=0.01

accumulate_grad_batches=2

batch_size=16
conv1_out_channels=32
conv2_out_channels=16

bottleneck_dim=32
num_lstm_layers=1
dropout=0.2

# lstm
model_name='lstm-encoder'

lstm_hidden_dim=$bottleneck_dim
lstm_num_layers=$num_lstm_layers
lstm_dropout=$dropout

for method in "${methods[@]}"; do
    for classifier in "${classifiers[@]}"; do
        for dataset in "${datasets[@]}"; do
            # if "category" is in name of dataset, then target_label is "category", else it is "device"
            if [[ $dataset == "dataset_a" ]]; then
                target_label="category"
                # if dataset is dataset_d1 or then the target label should be "class"
            # elif [[ $dataset == "dataset_d1" ]] || [[ $dataset == "dataset_d2" ]]; then
                # task="detection"
            else
                target_label="device"
            fi

            echo "=============================================="
            echo "=============================================="
            echo "Running classifier $classifier on dataset $dataset with target label $target_label using method $method"
                # --tuning \
                # --log \
            python classifier.py \
                --model_name=$model_name \
                --task=identification \
                --method=$method \
                --dataset=$dataset \
                --target_label=$target_label \
                --classifier=$classifier \
                --batch_size=$batch_size \
                --kfold=$folds \
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
                --early_stopping_patience=$early_stopping_patience \
                --learning_rate_patience=$learning_rate_patience \
                --base_model=$base_model \
                --use_batch_norm



        done
    done
done

# create tables
# python create_tables.py -i results
