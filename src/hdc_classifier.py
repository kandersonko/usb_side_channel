#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

import lightning.pytorch as pl
from config import default_config, merge_config_with_cli_args
from utils.data import load_data

from sklearn.metrics import classification_report

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
from torchmetrics import Accuracy
from tqdm import tqdm

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings

# HDC/VSA encoder
class SignalEncoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super().__init__()
        self.position = embeddings.Random(size, out_features)
        self.value = embeddings.Level(levels, out_features)

    def forward(self, x):
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)




# python ./hdc_classifier.py --dataset=dataset_a --target_label=category --batch_size=8 --dataset_subset=features --sequence_length=32

def main():
    config = merge_config_with_cli_args(default_config)

    pl.seed_everything(config['seed'])

    # load the dataset
    print("Loading the dataset")


    dataset_subset = config['dataset_subset']

    data_dir = config['data_dir']

    dataset_name = config['dataset']
    target_label = config['target_label']

    # dataset = np.load(f"{data_dir}/{subset}_dataset.npz", allow_pickle=True)
    dataset = None
    if dataset_subset == 'all':
        dataset = np.load(f"{data_dir}/{dataset_name}-{target_label}.npz", allow_pickle=True)
    elif dataset_subset == 'features':
        dataset = np.load(f"{data_dir}/{dataset_name}-{target_label}_features.npz", allow_pickle=True)
    else:
        raise ValueError(f"Unknown dataset subset {dataset_subset}")
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    target_names = dataset['target_names']
    print(f"Loaded dataset {dataset_name} with {len(target_names)} classes")
    print("dataset shape:", X_train.shape, X_val.shape, X_test.shape)

    num_classes = len(target_names)

    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())

    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)

    test_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())

    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

    # model
    dimensions = int(config.get('dimensions', 50000))
    num_levels = int(config.get('num_levels', 2000))
    encode = SignalEncoder(dimensions, config['sequence_length'], num_levels)
    model = Centroid(dimensions, num_classes)

    # training
    print("Training...")
    with torch.no_grad():
        # add multiple epochs training
        for epoch in tqdm(range(config['max_epochs']), desc="Training"):
            for samples, labels in train_ld:
                sample_hv = encode(samples)
                model.add_online(sample_hv, labels)

    print("Testing...")

    accuracy = Accuracy("multiclass", num_classes=num_classes)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        model.normalize()

        for samples, labels in tqdm(test_ld, desc="Testing"):

            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            accuracy.update(outputs, labels)

            # Store predictions and labels
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


        print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
        # Compute and print classification report
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=target_names))



if __name__ == '__main__':
    main()
