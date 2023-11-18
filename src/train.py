#!/usr/bin/env python3

import os

from sklearn.ensemble import RandomForestClassifier

import torch

import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
# from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
import wandb

from models.autoencoders import Autoencoder
from models.utils import evaluate_detection
from dataset import extract_segments, SegmentedSignalDataModule, encode_dataset_in_batches, extract_features
from config import default_config

from callbacks import InputMonitor

import argparse


torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn')

argparser = argparse.ArgumentParser()

# parse even unknown arguments

def merge_config_with_cli_args(config):
    # Create the argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--SEED', type=int, default=42)
    parser.add_argument('--BATCH_SIZE', type=int, default=512)
    parser.add_argument('--VAL_SPLIT', type=float, default=0.2)
    parser.add_argument('--BOTTLENECK_DIM', type=int, default=32)
    parser.add_argument('--MAX_EPOCHS', type=int, default=100)
    parser.add_argument('--MIN_EPOCHS', type=int, default=10)
    parser.add_argument('--EARLY_STOPPING_PATIENCE', type=int, default=10)
    parser.add_argument('--MONITOR_METRIC', type=str, default='val_loss')
    parser.add_argument('--CHECKPOINT_PATH', type=str, default='best_models/')
    parser.add_argument('--ACCUMULATE_GRAD_BATCHES', type=int, default=1)
    parser.add_argument('--NUM_WORKERS', type=int, default=0)
    parser.add_argument('--CONV1_OUT_CHANNELS', type=int, default=64)
    parser.add_argument('--CONV2_OUT_CHANNELS', type=int, default=128)
    # add segment overlap
    parser.add_argument('--OVERLAP', type=float, default=0.75)
    # add dropout
    parser.add_argument('--DROPOUT', type=float, default=0.2)

    # lstm number of layers
    parser.add_argument('--NUM_LAYERS', type=int, default=1)

    # add learning rate
    parser.add_argument('--LEARNING_RATE', type=float, default=1e-3)
    # add reconstruction loss weight
    parser.add_argument('--RECONSTRUCTION_LOSS_WEIGHT', type=float, default=1.0)
    # add class loss weight
    parser.add_argument('--CLASS_LOSS_WEIGHT', type=float, default=1.0)

    # add dataset subset
    parser.add_argument('--DATASET_SUBSET', type=str, default='all')
    # add dataset path
    parser.add_argument('--DATASET_PATH', type=str, default='data/datasets.pkl')




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

def main():

    # initialize wandb only on the main process

    # merge the command line arguments with the config dictionary
    # make sure args is not empty
    # config = {**default_config, **vars(args)} if vars(args) else default_config

    config = merge_config_with_cli_args(default_config)


    wandb_logger = WandbLogger(project="USB", log_model="all", config=config)
    # wandb_logger = WandbLogger(project="USB", config=config)
    # wandb_logger.watch(model)

    wandb.init(project='USB')

    # Update config dictionary with values from wandb.config if available
    for key in config.keys():
        if key in wandb.config:
            config[key] = wandb.config[key]


    # seed everything
    pl.seed_everything(config['SEED'])

    # model = LSTMAutoencoder(config['BOTTLENECK_DIM'])
    model = Autoencoder(bottleneck_dim=config['BOTTLENECK_DIM'])
    summary = ModelSummary(model, max_depth=-1)


    early_stopping = EarlyStopping(config['MONITOR_METRIC'], patience=config['EARLY_STOPPING_PATIENCE'], verbose=False, mode='min', min_delta=0.0)
    learning_rate_monitor = LearningRateMonitor(logging_interval='epoch', log_momentum=True)

    # input_monitor = InputMonitor()

    checkpoint_callback = ModelCheckpoint(
        monitor=config['MONITOR_METRIC'],  # or another metric such as 'val_accuracy'
        dirpath=config['CHECKPOINT_PATH'],
        filename='best_model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',  # 'min' for loss and 'max' for accuracy
    )

    learning_rate_finder = LearningRateFinder()


    callbacks = [
        early_stopping,
        learning_rate_monitor,
        checkpoint_callback,
        learning_rate_finder,
    ]

    # data_module = USBDataModule(batch_size=config['BATCH_SIZE'], val_split=config['VAL_SPLIT'])
    data_module = SegmentedSignalDataModule(batch_size=config['BATCH_SIZE'], val_split=config['VAL_SPLIT'])

    torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(
        # accumulate_grad_batches=config['ACCUMULATE_GRAD_BATCHES'],
        num_sanity_val_steps=0,
        max_epochs=config['MAX_EPOCHS'],
        min_epochs=config['MIN_EPOCHS'],
        accelerator="gpu",
        devices=-1,
        strategy='ddp',
        logger=wandb_logger,
        callbacks=callbacks,
        precision="32-true",
        # precision="16-mixed",
        # precision=32,
        # default_root_dir=config['CHECKPOINT_PATH'],
    )

    # # learning rate finder
    # if trainer.is_global_zero:
    #     print("Finding the learning rate")

    # tuner = Tuner(trainer)
    # tuner.lr_find(model, num_training=100, datamodule=data_module, max_lr=0.1)

    # if trainer.is_global_zero:
    #     # print config
    #     print("Config:")
    #     for key in config.keys():
    #         print(key, ":", config[key])
    #     print()


    if trainer.is_global_zero:
        print(model)
        print(summary)

    # trainer.tune(model, datamodule=data_module)

    # data_module = SegmentedSignalDataModule(batch_size=config['BATCH_SIZE'], val_split=config['VAL_SPLIT'])

    trainer.fit(model, datamodule=data_module)

    # feature extraction

    # Load the best model
    # model = ConvAutoencoder.load_from_checkpoint(early_stopping.best_model_path)

    # Load the best model
    best_model_path = checkpoint_callback.best_model_path
    # model = Autoencoder.load_from_checkpoint(best_model_path)
    model.load_state_dict(torch.load(best_model_path)['state_dict'])


    if trainer.is_global_zero:
        print("Setting up the dataset")

    data_module = SegmentedSignalDataModule(batch_size=config['BATCH_SIZE'])
    data_module.setup()

    if trainer.is_global_zero:
        print("Extracting the segments")
    # Extract segments and labels from the training dataset
    X_train, y_train, X_test, y_test = extract_segments(data_module)

    target_names = data_module.target_names

    if trainer.is_global_zero:
        print("Evaluating the model")

    # Determine the process rank using the environment variable
    if trainer.is_global_zero:
        # training a random forest classifier without feature extraction
        classifier = RandomForestClassifier(max_depth=10, random_state=42, n_jobs=-1)
        accuracy, report = evaluate_detection(classifier, X_train, y_train, X_test, y_test, target_names)

        print("dataset shape: ", X_train.shape, y_train.shape)

        # log the results to wandb
        wandb.log({"identification accuracy (random forest no feature extraction)": accuracy})


        print()
        print("Without feature extraction")
        print("Classifier: ", classifier.__class__.__name__)
        print(f"Accuracy: {accuracy*100.0:.4f}")
        print(report)
        print()


    # # training a random forest classifier with feature extraction
    # X_train_encoded = encode_dataset_in_batches(model, torch.tensor(X_train, dtype=torch.float32))
    # X_test_encoded = encode_dataset_in_batches(model, torch.tensor(X_test, dtype=torch.float32))

    if trainer.is_global_zero:
        print("Extracting features")
    X_train_encoded, y_train, X_test_encoded, y_test = extract_features(model, data_module)

    if trainer.is_global_zero:
        print("Training the classifier")

    if trainer.is_global_zero:
        classifier = RandomForestClassifier(max_depth=10, random_state=42, n_jobs=-1)

        accuracy, report = evaluate_detection(classifier, X_train_encoded, y_train, X_test_encoded, y_test, target_names)

        # log the results to wandb
        wandb.log({"identification accuracy (random forest with feature extraction)": accuracy})

        print("With feature extraction")
        print("Classifier: ", classifier.__class__.__name__)
        print(f"Accuracy: {accuracy*100.0:.4f}")
        print(report)


    # Close wandb run
    wandb.finish()

if __name__ == '__main__':
    main()
