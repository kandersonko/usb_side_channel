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

def merge_config_with_cli_args(config):
    # Create the argument parser
    parser = argparse.ArgumentParser()

    num_workers = cpu_count() // 2

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--val_split', type=float, default=0.2)
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
    parser.add_argument('--overlap', type=float, default=0.75)
    # add dropout
    parser.add_argument('--dropout', type=float, default=0.2)

    # lstm number of layers
    parser.add_argument('--num_layers', type=int, default=1)

    # add learning rate
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    # add reconstruction loss weight
    parser.add_argument('--reconstruction_loss_weight', type=float, default=1.0)
    # add class loss weight
    parser.add_argument('--classification_loss_weight', type=float, default=1.0)

    # add dataset subset
    parser.add_argument('--dataset_subset', type=str, default='all')
    # add dataset path
    parser.add_argument('--dataset_path', type=str, default='data/datasets.pkl')

    # add target label with some specific choices
    parser.add_argument('--target_label', type=str, default='category', choices=['category', 'class', 'state'])

    # add number of classes
    parser.add_argument('--num_classes', type=int, default=5)

    # add model path
    parser.add_argument('--model_path', type=str, default='')


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
