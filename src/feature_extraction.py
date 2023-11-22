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

from models.autoencoders import Autoencoder
from models.utils import evaluate_detection
from dataset import extract_segments, SegmentedSignalDataModule, encode_dataset_in_batches, extract_features, setup_dataset

from config import default_config, merge_config_with_cli_args


# from callbacks import InputMonitor

# path the best model path as an argument

torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    config = merge_config_with_cli_args(default_config)

    if config['model_path'] is None:
        raise ValueError("Provide a model path")

    task = config.get('task', None)
    if task is None:
        raise ValueError("Provide a task (identification or detection))")
    # config['batch_size'] = 512


    pl.seed_everything(config['seed'], workers=True)

    # plot data
    plot_data = []


    print("Setting up the model")

    # best_model_path = checkpoint_callback.best_model_path
    # best_model_path = 'best_models/best_model-epoch=16-val_loss=0.36.ckpt'
    # best_model_path = 'best_models/best_model-epoch=49-val_loss=0.11.ckpt'
    best_model_path = config['model_path']

    model = Autoencoder(**config)
    summary = ModelSummary(model, max_depth=-1)
    print(model)
    print(summary)

    model.load_state_dict(torch.load(best_model_path)['state_dict'])


    if task == "identification":
        config['target_label'] = "category"
        config['num_classes'] = 5
        title = 'Identification'
    elif task == "detection":
        config['target_label'] = "class"
        config['num_classes'] = 2
        title = 'Anomaly Detection'
    else:
        raise ValueError("Provide a valid task")

    # measure time for data setup
    start_time = time.time()

    print("Setting up the dataset")
    # use the setup_dataset function
    signals, labels, target_names = setup_dataset(**config)
    # convert to tensors
    signals = torch.from_numpy(signals).float()
    labels = torch.from_numpy(labels).long()
    # create data loaders
    dataset = TensorDataset(signals, labels)
    train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * (1 - config['val_split'])), int(len(dataset) * config['val_split'])])
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])


    end_time = time.time()
    duration = end_time - start_time
    plot_data.append({'name': 'dataset', 'task': 'setup', 'dataset': 'all', 'duration': duration})

    # extract semgnets
    data_module = SegmentedSignalDataModule(**config)
    data_module.setup()
    X_train, y_train, X_test, y_test = extract_segments(data_module)

    end_time = time.time()
    duration = end_time - start_time
    plot_data.append({"name": "dataset", "task": "extract segments", 'dataset': 'all', "duration": duration})

    raw_target_names = data_module.target_names

    # save the segments and labels dataset to disk to the data/ folder
    # using numpy
    target_label = config['target_label']
    np.save(f"data/X_{target_label}_train_raw.npy", X_train)
    np.save(f"data/y_{target_label}_train_raw.npy", y_train)
    np.save(f"data/X_{target_label}_test_raw.npy", X_test)
    np.save(f"data/y_{target_label}_test_raw.npy", y_test)
    # save the target names
    np.save(f"data/{target_label}_target_names_raw.npy", raw_target_names)



    print("Extracting features")
    start_time = time.time()
    X_train_encoded, y_train, X_test_encoded, y_test = extract_features(
        model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    end_time = time.time()
    duration = end_time - start_time
    plot_data.append({"name": "autoencoder", "task": "extract features", "dataset": "all", "duration": duration})
    # save the plot data
    plot_data = pd.DataFrame(plot_data)
    plot_data.to_csv('data/feature_extraction_plot_data.csv')


    # save the extracted features
    print("Saving the features")
    target_label = config['target_label']
    np.save(f"data/X_{target_label}_train_encoded.npy", X_train_encoded)
    np.save(f"data/y_{target_label}_train_encoded.npy", y_train)
    np.save(f"data/X_{target_label}_test_encoded.npy", X_test_encoded)
    np.save(f"data/y_{target_label}_test_encoded.npy", y_test)
    np.save(f"data/{target_label}_target_names_encoded.npy", target_names)




if __name__ == '__main__':
    main()
