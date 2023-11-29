
import os

import torch

import numpy as np
from sklearn.model_selection import train_test_split


import lightning.pytorch as pl
from dataset import extract_segments, SegmentedSignalDataModule, encode_dataset_in_batches, extract_features
from config import default_config, merge_config_with_cli_args

def save_dataset(target_label, seed, config):

    # seed everything
    pl.seed_everything(seed)

    # setup the dataset
    data_module = SegmentedSignalDataModule(**config)
    data_module.setup()

    target_names = data_module.target_names

    # extract the segments
    X_train, y_train, X_test, y_test = extract_segments(data_module)

    # split the dataset into train, val and test
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=seed
    )

    # save the dataset to disk
    np.savez_compressed(
        os.path.join(config['data_dir'], f'{target_label}_dataset.npz'),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        target_names=target_names,
    )


def main():

    config = merge_config_with_cli_args(default_config)

    # seed everything
    seed = config['seed']
    pl.seed_everything(seed)

    # save category dataset
    print("Saving the category dataset")
    config['target_label'] = "category"
    config['num_classes'] = 5
    target_label = config['target_label']
    save_dataset(target_label, seed, config)

    # save class dataset
    print("Saving the class dataset")
    config['target_label'] = "class"
    config['num_classes'] = 2
    target_label = config['target_label']
    save_dataset(target_label, seed, config)



if __name__ == '__main__':
    main()
