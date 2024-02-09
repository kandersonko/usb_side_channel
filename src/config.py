import argparse
import yaml

from multiprocessing import cpu_count

global default_config


# Load the configuration from the YAML file
def load_config():
    with open('config.yaml', 'r') as yaml_file:
        return yaml.safe_load(yaml_file)


default_config = load_config()

argparser = argparse.ArgumentParser()

# parse even unknown arguments
# python inference.py --model_path=best_models/best_model-epoch=42-val_loss=0.02.ckpt --batch_size=267 --bottleneck_dim=355 --classification_loss_weight=0.2354142206693032 --conv1_out_channels=308 --conv2_out_channels=90 --dropout=0.638829782779092 --num_lstm_layers=6 --reconstruction_loss_weight=0.8200835209023157


def merge_config_with_cli_args(config):
    # Create the argument parser
    parser = argparse.ArgumentParser()

    num_workers = cpu_count() // 2

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--bottleneck_dim', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--min_epochs', type=int, default=10)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    # add learning rate patience
    parser.add_argument('--learning_rate_patience', type=int, default=3)
    parser.add_argument('--monitor_metric', type=str, default='val_loss')
    parser.add_argument('--checkpoint_path', type=str, default='best_models/')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=num_workers)
    parser.add_argument('--conv1_out_channels', type=int, default=64)
    parser.add_argument('--conv2_out_channels', type=int, default=128)
    # add segment overlap
    parser.add_argument('--overlap', type=float, default=0.0)
    # add sequence length
    parser.add_argument('--sequence_length', type=int, default=10000)
    # add dropout
    parser.add_argument('--dropout', type=float, default=0.2)

    # lstm number of layers
    parser.add_argument('--num_lstm_layers', type=int, default=1)

    # add learning rate
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    # add reconstruction loss weight
    parser.add_argument('--reconstruction_loss_weight',
                        type=float, default=1.0)
    # add class loss weight
    parser.add_argument('--classification_loss_weight',
                        type=float, default=1.0)

    # # add no_class_weights boolean flag with default False
    # parser.add_argument('--use_class_weights', action='store_true', default=False)

    # add log bool argument
    parser.add_argument('--log', action='store_true', default=False)
    # add dataset name argument
    parser.add_argument('--dataset', type=str, default='dataset_a')
    # add dataset subset
    parser.add_argument('--dataset_subset', type=str, default='all')
    # add dataset path
    parser.add_argument('--dataset_path', type=str,
                        default='data/datasets.pkl')

    # add max_samples_per_class
    parser.add_argument('--max_samples_per_class', type=int, default=1000)

    # add kfold
    parser.add_argument('--kfold', type=int, default=10)

    # data_dir for saving the dataset
    parser.add_argument('--data_dir', type=str, default='datasets')

    # add target label with some specific choices
    parser.add_argument('--target_label', type=str,
                        default='category', choices=['category', 'class', 'state', 'device', 'device_name', 'brand'])

    # add number of classes
    parser.add_argument('--num_classes', type=int, default=5)

    # add model path
    parser.add_argument('--model_path', type=str, default='')

    # add workers
    parser.add_argument('--workers', type=int, default=4)

    # add memory
    parser.add_argument('--memory', type=str, default='4GB')

    # add chunk_size
    parser.add_argument('--chunk_size', type=int, default=1000)

    # add use_local_cluster
    parser.add_argument('--use_local_cluster', action='store_true', default=True)

    # Use parse_known_args to accept arbitrary arguments
    args, unknown_args = parser.parse_known_args()

    # Convert args to dictionary
    cli_args = vars(args)

    # Handle unknown arguments (optional)
    for arg in unknown_args:
        if arg.startswith('--'):
            key, value = arg.lstrip('--').split('=')
            cli_args[key] = value

    # Merge the CLI arguments into the config dictionary
    config.update(cli_args)

    return config
