from pathlib import Path
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger

from models.autoencoders import Autoencoder, PureAutoencoder
from models.utils import evaluate_detection
from dataset import extract_segments, SegmentedSignalDataModule, encode_dataset_in_batches, extract_features, setup_dataset
from dataset import to_dataloader

from config import default_config, merge_config_with_cli_args


# from callbacks import InputMonitor

# path the best model path as an argument

torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    config = merge_config_with_cli_args(default_config)

    benchmarking = config.get('benchmarking', False)

    if config['model_path'] is None:
        raise ValueError("Provide a model path")

    pl.seed_everything(config['seed'], workers=True)

    subset = config['dataset_subset']

    # plot data
    plot_data = []

    print("Setting up the model")

    best_model_path = config['model_path']
    print(f"Loading the model from {best_model_path}")

    model = None
    if config['model_path'] is not None and config['model_path'] != '':
        model = PureAutoencoder.load_from_checkpoint(best_model_path)
    else:
        raise ValueError("Provide a model path")

    summary = ModelSummary(model, max_depth=-1)
    print(model)
    print(summary)

    data_dir = config['data_dir']

    print("Loading the dataset")
    target_label = config['target_label']
    dataset_name = config['dataset']
    dataset = np.load(f"{data_dir}/{dataset_name}-{target_label}.npz", allow_pickle=True)
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    target_names = dataset['target_names']

    device = torch.device("cpu")
    if not benchmarking:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            model = model.to(device)

    else:
        model = model.cpu()


    train_loader = to_dataloader(X_train, y_train, batch_size=config['batch_size'],
                                    num_workers=config['num_workers'], shuffle=True)
    val_loader = to_dataloader(X_val, y_val, batch_size=config['batch_size'],
                                    num_workers=config['num_workers'], shuffle=False)
    test_loader = to_dataloader(X_test, y_test, batch_size=config['batch_size'],
                                    num_workers=config['num_workers'], shuffle=False)

    print("Extracting features")
    start_time = time.time()

    X_train_encoded, y_train, X_val_encoded, y_val, X_test_encoded, y_test = extract_features(
        model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        has_modules=True,
    )

    end_time = time.time()
    duration = end_time - start_time
    plot_data.append({"name": "autoencoder", "task": "extract features", "dataset": f"{dataset_name}", "duration": duration})
    filename = Path(f'measurements/{dataset_name}-{subset}-autoencoder-feature-extraction-duration.csv')

    # save the plot data
    plot_data = pd.DataFrame(plot_data)
    plot_data.to_csv(filename)

    data_dir = config['data_dir']

    # save the extracted features
    if not benchmarking:
        print("Saving the features")
        np.savez_compressed(
            f"{data_dir}/{dataset_name}-{target_label}_features.npz",
            X_train=X_train_encoded,
            y_train=y_train,
            X_val=X_val_encoded,
            y_val=y_val,
            X_test=X_test_encoded,
            y_test=y_test,
            target_names=target_names,
        )



if __name__ == '__main__':
    main()
