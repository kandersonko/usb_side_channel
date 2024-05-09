#!/usr/bin/env python3

import os

import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

import torch

import lightning.pytorch as pl
# from lightning.pytorch.tuner import Tuner
# from lightning.pytorch.loggers import TensorBoardLogger
# from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
# from lightning.pytorch.callbacks import LearningRateFinder, Timer
# from lightning.pytorch.utilities.model_summary import ModelSummary
# from lightning.pytorch.cli import LightningCLI
# from lightning.pytorch.loggers import WandbLogger
# import wandb

from models.autoencoders import Autoencoder
# from models.autoencoders import PureAutoencoder
from models.utils import evaluate_detection
from dataset import extract_segments, SegmentedSignalDataModule, to_dataloader
from dataset import extract_features, get_dataloaders, encode_dataset_in_batches
# from config import default_config, merge_config_with_cli_args
import pandas as pd

import hydra
from omegaconf import DictConfig, OmegaConf

# from callbacks import InputMonitor


from utilities import RankedLogger, task_wrapper, instantiate_callbacks, instantiate_loggers

log = RankedLogger(__name__, rank_zero_only=True)


torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method('spawn')



@task_wrapper
def train(config: DictConfig):

    logger = instantiate_loggers(config.logger)

    # seed everything
    pl.seed_everything(config.seed)

    # the training dataset
    train_dataset_path = f"{config.datasets.data_dir}/common_dataset.npz"
    training_dataset = np.load(train_dataset_path, allow_pickle=True)
    X_train = training_dataset['X_train']
    y_train = training_dataset['y_train']
    X_val = training_dataset['X_val']
    y_val = training_dataset['y_val']
    target_names = training_dataset['target_names']
    num_classes = len(target_names)

    # get the data loaders
    train_loader = to_dataloader(X_train, y_train, batch_size=config.datasets.batch_size,
                                 num_workers=config.datasets.num_workers, shuffle=True)
    val_loader = to_dataloader(X_val, y_val, batch_size=config.datasets.batch_size,
                                 num_workers=config.datasets.num_workers, shuffle=False)

    # update the number of classes in the config
    config.datasets.num_classes = num_classes
    log.info(f"Number of classes: {config.datasets.num_classes}")
    log.info(f"Number of training samples: {len(X_train)}")
    log.info(f"Number of validation samples: {len(X_val)}")
    log.info(f"Number of features: {X_train.shape[1]}")

    model = Autoencoder(**config.model)
    # model = PureAutoencoder(**config)

    # summary = ModelSummary(model, max_depth=-1)

    # early_stopping = EarlyStopping(
    #     config.callback.early_stopping.monitor, patience=config.callback.early_stopping.patience, verbose=False, mode=config.callback.early_stopping.mode, min_delta=config.callback.early_stopping.min_delta)
    # learning_rate_monitor = LearningRateMonitor(
    #     logging_interval='epoch', log_momentum=True)

    # # input_monitor = InputMonitor()
    # date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")

    # checkpoint_callback = ModelCheckpoint(
    #     # or another metric such as 'val_accuracy'
    #     monitor=config.callback.model_checkpoint.monitor,
    #     dirpath=config.callback.model_checkpoint.dirpath,
    #     filename='autoencoder-{epoch:02d}-{val_loss:.2f}-'+date_time,
    #     save_top_k=1,
    #     mode='min',  # 'min' for loss and 'max' for accuracy
    # )

    # learning_rate_finder = LearningRateFinder()

    # # stop training after 12 hours
    # timer = Timer(duration="00:12:00:00")

    # callbacks = [
    #     early_stopping,
    #     learning_rate_monitor,
    #     checkpoint_callback,
    #     learning_rate_finder,
    #     timer,
    # ]

    # data_module = USBDataModule(batch_size=config['BATCH_SIZE'], val_split=config['VAL_SPLIT'])

    callbacks = instantiate_callbacks(config.callbacks)


    torch.set_float32_matmul_precision('medium')

    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)
    # trainer = pl.Trainer(
    #     deterministic=True, # makes training reproducible
    #     accumulate_grad_batches=config.get('accumulate_grad_batches'),
    #     num_sanity_val_steps=0,
    #     max_epochs=config['max_epochs'],
    #     min_epochs=config['min_epochs'],
    #     accelerator="gpu",
    #     devices=-1,
    #     strategy='ddp',
    #     logger=logger,
    #     callbacks=callbacks,
    #     precision="32-true",
    #     # precision="16-mixed",
    #     # precision=32,
    #     # default_root_dir=config['CHECKPOINT_PATH'],
    # )

    # # learning rate finder
    # if trainer.is_global_zero:
    #     log.info("Finding the learning rate")

    # tuner = Tuner(trainer)
    # tuner.lr_find(model, num_training=100, datamodule=data_module, max_lr=0.1)

    # if trainer.is_global_zero:
    log.info(model)
    # log.info(summary)

    # trainer.tune(model, datamodule=data_module)

    # data_module = SegmentedSignalDataModule(batch_size=config['BATCH_SIZE'], val_split=config['VAL_SPLIT'])

    trainer.fit(model, train_loader, val_loader)

    log.info("Finished training")
    log.info("callback metrics: %s", trainer.callback_metrics)

    log.info("trainer: %s", dir(trainer))

    # timer = trainer.timer_callback.timers
    timer = None
    for callback in trainer.callbacks:
        if isinstance(callback, pl.callbacks.Timer):
            timer = callback

    duration = timer.time_elapsed("train")
    log.info(f"Training duration: {duration}")
    # save the training duration
    plot_data = []
    plot_data.append({"name": "autoencoder", "task": "training", "dataset": "all_training_data", "method": "autoencoder", "duration": duration})

    plot_data = pd.DataFrame(plot_data)

    if trainer.is_global_zero:

        # Load the best model
        best_model_path = trainer.checkpoint_callback.best_model_path
        # model = PureAutoencoder.load_from_checkpoint(best_model_path)
        model = Autoencoder.load_from_checkpoint(best_model_path)
        # model.load_state_dict(torch.load(best_model_path)['state_dict'])


        date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
        plot_data.to_csv(f"measurements/autoencoder-{date_time}-training_duration.csv", index=False)

        # if trainer.is_global_zero:
        log.info("\nFinished training\n\n")
        log.info("Evaluating the model")
        log.info(f"Best model path: {best_model_path}")
        log.info(f"Dataset:  {config.datasets.dataset}")
        log.info(f"Target label: {config.datasets.target_label}")


        # the evaluation dataset
        dataset_path = f"{config.datasets.data_dir}/{config.datasets.dataset}-{config.datasets.target_label}.npz"
        dataset = np.load(dataset_path, allow_pickle=True)
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_val = dataset['X_val']
        y_val = dataset['y_val']
        X_test = dataset['X_test']
        y_test = dataset['y_test']
        target_names = dataset['target_names']

        (train_loader, val_loader, test_loader), class_weights = get_dataloaders(
            X_train, y_train, X_val, y_val, X_test, y_test,
            num_workers=config.datasets.num_workers,
            batch_size=config.datasets.batch_size,
        )
        # Determine the process rank using the environment variable
        # training a random forest classifier without feature extraction
        classifier = RandomForestClassifier(
            max_depth=10, random_state=42, n_jobs=-1)
        accuracy, report = evaluate_detection(
            classifier, X_train, y_train, X_test, y_test, target_names)

        log.info(f"dataset shape: {X_train.shape}, {y_train.shape}")

        # log the results to wandb
        # wandb.log(
        # {"identification accuracy (random forest no feature extraction)": accuracy})

        log.info("Without feature extraction")
        log.info(f"Classifier:  classifier.__class__.__name__")
        log.info(f"Accuracy: {accuracy*100.0:.4f}")
        log.info(report)

        # # training a random forest classifier with feature extraction
        # X_train_encoded = encode_dataset_in_batches(model, torch.tensor(X_train, dtype=torch.float32))
        # X_test_encoded = encode_dataset_in_batches(model, torch.tensor(X_test, dtype=torch.float32))

        # if trainer.is_global_zero:
        log.info("Extracting features")
        X_train_encoded, y_train, X_val_encoded, y_val, X_test_encoded, y_test = extract_features(
            model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            has_modules=False
        )

        # if trainer.is_global_zero:
        log.info(f"dataset shape: {X_train_encoded.shape}, {y_train.shape}")
        log.info("Training the classifier")

        # if trainer.is_global_zero:
        classifier = RandomForestClassifier(
            max_depth=10, random_state=42, n_jobs=-1)

        accuracy, report = evaluate_detection(
            classifier, X_train_encoded, y_train, X_test_encoded, y_test, target_names)

            # log the results to wandb
            # wandb.log(
            #     {"identification accuracy (random forest with feature extraction)": accuracy})

        log.info("With feature extraction")
        log.info(f"Classifier: ", classifier.__class__.__name__)
        log.info(f"Accuracy: {accuracy*100.0:.4f}")
        log.info(report)

        log.info(f"Best model path: {best_model_path}")


    # merge train and test metrics
    task_metrics = trainer.callback_metrics
    metric_dict = {**task_metrics, **task_metrics}
    object_dict = {"model": model, "trainer": trainer, "config": config, "callbacks": callbacks, "timer": timer}
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="./configs", config_name="config.yaml")
def main(config: DictConfig):
    log.info(OmegaConf.to_yaml(config))
    train(config)

if __name__ == '__main__':
    main()
