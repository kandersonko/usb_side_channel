#!/usr/bin/env python3

import argparse
import wandb
import json
import numpy as np
import pandas as pd

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from classifier import tune
from train import train
from config import default_config, merge_config_with_cli_args


parser = argparse.ArgumentParser()
# add wandb run path
parser.add_argument('--wandb_path', type=str, help="Wandb run or sweep path", required=True)
# add is_sweep boolean flag
parser.add_argument('--is_sweep', action='store_true', default=False)
# add resume boolean flag
parser.add_argument('--resume', action='store_true', default=False)

args = parser.parse_args()


def get_dataset_subsets(dataset):
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_val = dataset['X_val']
        y_val = dataset['y_val']
        X_test = dataset['X_test']
        y_test = dataset['y_test']
        target_names = dataset['target_names']
        return X_train, y_train, X_val, y_val, X_test, y_test, target_names


def load_dataset(data_dir, dataset_name, target_label, method):
    dataset_path = None
    X_train, y_train, X_val, y_val, X_test, y_test, target_names = None, None, None, None, None, None, None
    if method == 'tsfresh':
        dataset_path = f"{data_dir}/{dataset_name}-{target_label}.npz"
        raw_dataset = np.load(dataset_path, allow_pickle=True)

        train = pd.read_csv(f"{data_dir}/{dataset_name}-{target_label}-train_tsfresh.csv")
        y_train = train[target_label].values
        X_train = train.drop(columns=[target_label], axis=1).values

        val = pd.read_csv(f"{data_dir}/{dataset_name}-{target_label}-val_tsfresh.csv")
        y_val = val[target_label].values
        X_val = val.drop(columns=[target_label], axis=1).values

        test = pd.read_csv(f"{data_dir}/{dataset_name}-{target_label}-test_tsfresh.csv")
        y_test = test[target_label].values
        X_test = test.drop(columns=[target_label], axis=1).values

        target_names = raw_dataset['target_names']

    elif method == 'encoder':
        dataset_path = f"{data_dir}/{dataset_name}-{target_label}_features.npz"
        dataset = np.load(dataset_path, allow_pickle=True)
        X_train, y_train, X_val, y_val, X_test, y_test, target_names = get_dataset_subsets(dataset)

    elif method == 'raw':
        dataset_path = f"{data_dir}/{dataset_name}-{target_label}.npz"
        dataset = np.load(dataset_path, allow_pickle=True)
        X_train, y_train, X_val, y_val, X_test, y_test, target_names = get_dataset_subsets(dataset)

    else:
        raise ValueError(f"Invalid method: {method}. Must be one of 'tsfresh', 'encoder', or 'raw'.")

    return X_train, y_train, X_val, y_val, X_test, y_test, target_names


def main():
    wandb.login()
    api = wandb.Api()
    is_sweep = args.is_sweep
    wandb_path = args.wandb_path
    best_run = None
    if is_sweep:
        sweep = api.sweep(wandb_path)
        best_run = sweep.best_run()
    else:
        best_run = api.run(wandb_path)

    # load config from wandb api
    print("Loading config from wandb")
    cli_config = merge_config_with_cli_args(default_config)
    # merge the best run config with the cli config with the best run config taking precedence
    config = {**cli_config, **best_run.config}
    # config = best_run.config
    print(config)

    pl.seed_everything(config['seed'])


    # load dataset
    # print(f"Loading dataset: {config['dataset']}")
    # X_train, y_train, X_val, y_val, X_test, y_test, target_names = load_dataset(config['data_dir'], config['dataset'], config['target_label'], config['method'])
    # print("X_train shape: ", X_train.shape)
    # print("y_train shape: ", y_train.shape)
    # print("X_val shape: ", X_val.shape)
    # print("y_val shape: ", y_val.shape)
    # print("X_test shape: ", X_test.shape)
    # print("y_test shape: ", y_test.shape)

    wandb_id = wandb_path.split('/')[-1]
    print("wandb_id: ", wandb_id)

    train(config)

    if cli_config.get('tuning', False):
        print("Tuning the model")
        if args.resume:
            print("Resuming the tuning")
            wandb.init(project="usb_side_channel", config=config, resume=wandb_id)
        # else:
            # wandb.init(project="usb_side_channel", config=config)

        # wandb_logger = WandbLogger(project="usb_side_channel", config=config)
        # logger = wandb_logger

        # config['logger'] = logger
        task = config['task']
        tune(config, X_train, y_train, X_val, y_val, X_test, y_test, task, target_names)

    # elif cli_config('train'):
    #     train(config)
    #     print("Training the model")
        # wandb.init(project="usb_side_channel", config=config)
        # wandb_logger = WandbLogger(project="usb_side_channel", config=config)
        # logger = wandb_logger



if __name__ == '__main__':
    main()
