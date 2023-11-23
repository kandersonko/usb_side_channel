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
from config import default_config, merge_config_with_cli_args

from callbacks import InputMonitor

import argparse


torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn')


def main():

    # initialize wandb only on the main process

    # merge the command line arguments with the config dictionary
    # make sure args is not empty
    # config = {**default_config, **vars(args)} if vars(args) else default_config


    config = merge_config_with_cli_args(default_config)

    wandb_logger = WandbLogger(project="usb_side_channel", log_model="all", config=config)
    # wandb_logger = WandbLogger(project="usb_side_channel", config=config)
    # wandb_logger.watch(model)

    # wandb.init(project='usb_side_channel', config=config)

    # # Update config dictionary with values from wandb.config if available
    # for key in config.keys():
    #     if key in wandb.config:
    #         config[key] = wandb.config[key]

    # seed everything
    pl.seed_everything(config['seed'])

    data_module = SegmentedSignalDataModule(**config)
    data_module.setup()

    class_weights = data_module.class_weights


    model = None
    if config['use_class_weights']:
        model = Autoencoder(**config, class_weights=class_weights)

        print("class weights: ", class_weights)
    else:
        model = Autoencoder(**config)
    summary = ModelSummary(model, max_depth=-1)

    print("class weights: ", class_weights)
    early_stopping = EarlyStopping(
        config['monitor_metric'], patience=config['early_stopping_patience'], verbose=False, mode='min', min_delta=0.0)
    learning_rate_monitor = LearningRateMonitor(
        logging_interval='epoch', log_momentum=True)

    # input_monitor = InputMonitor()

    checkpoint_callback = ModelCheckpoint(
        # or another metric such as 'val_accuracy'
        monitor=config['monitor_metric'],
        dirpath=config['checkpoint_path'],
        filename='best_model-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}',
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

    torch.set_float32_matmul_precision('medium')

    trainer = pl.Trainer(
        # accumulate_grad_batches=config['ACCUMULATE_GRAD_BATCHES'],
        num_sanity_val_steps=0,
        max_epochs=config['max_epochs'],
        min_epochs=config['min_epochs'],
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
    model = Autoencoder.load_from_checkpoint(best_model_path)
    # model.load_state_dict(torch.load(best_model_path)['state_dict'])

    if trainer.is_global_zero:
        print("Setting up the dataset")

    data_module = SegmentedSignalDataModule(**config)
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
        classifier = RandomForestClassifier(
            max_depth=10, random_state=42, n_jobs=-1)
        accuracy, report = evaluate_detection(
            classifier, X_train, y_train, X_test, y_test, target_names)

        print("dataset shape: ", X_train.shape, y_train.shape)

        # log the results to wandb
        # wandb.log(
        # {"identification accuracy (random forest no feature extraction)": accuracy})

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
    X_train_encoded, y_train, X_test_encoded, y_test = extract_features(
        model, data_module)

    if trainer.is_global_zero:
        print("Training the classifier")

    if trainer.is_global_zero:
        classifier = RandomForestClassifier(
            max_depth=10, random_state=42, n_jobs=-1)

        accuracy, report = evaluate_detection(
            classifier, X_train_encoded, y_train, X_test_encoded, y_test, target_names)

        # log the results to wandb
        # wandb.log(
        #     {"identification accuracy (random forest with feature extraction)": accuracy})

        print("With feature extraction")
        print("Classifier: ", classifier.__class__.__name__)
        print(f"Accuracy: {accuracy*100.0:.4f}")
        print(report)

    # Close wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
