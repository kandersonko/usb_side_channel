from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

import lightning.pytorch as pl
from multiprocessing import cpu_count


from utils.data import load_data

from config import default_config as config
from torch.utils.data import DataLoader
from torch.nn import DataParallel

def compute_class_weights(labels):
    """
    Compute class weights given a list/array of class labels.

    :param labels: An array-like object containing class labels.
    :return: A tensor of class weights.
    """
    # Count the number of instances of each class
    class_counts = torch.bincount(torch.tensor(labels, dtype=torch.long))

    # Compute weights (inverse frequency)
    weights = 1. / class_counts.float()

    # Normalize weights (this step is optional)
    weights = weights / weights.sum() * len(class_counts)

    return weights

def segment_dataset(features, labels, window_size, overlap):
    """
    Segments a dataset into windows of size window_size with overlap overlap
    :param features: The features to segment
    :param labels: The labels to segment
    :param window_size: The size of the window to apply to each signal
    :param overlap: The percentage of overlap between windows within the same signal
    :return: A tuple containing the segmented features and labels
    """
    assert 0 <= overlap < 1, "Overlap percent must be between 0 and 1"


    step_size = window_size - int(window_size * overlap)
    windows = []
    window_labels = []
    for feature, label in zip(features, labels):
        # Ensure each signal starts a new segmentation
        start = 0
        feature_length = len(feature)
        while start + window_size <= feature_length:
            end = start + window_size
            windows.append(feature[start:end])
            window_labels.append(label)
            start += step_size

    return windows, window_labels

def setup_dataset(dataset_path, sequence_length, val_split, dataset_subset, target_label, overlap, **kwargs):
    data = load_data(dataset_path)

    data['class'] = 'normal'
    data['class'] = data['class'].mask(data['brand'] == 'OMG', 'anomaly')
    data['class'] = data['class'].mask(data['brand'] == 'teensyduino', 'anomaly')

    # make the length of the data equal to 100k
    data['data'] = data['data'].map(lambda x: x[:len(x)-4])

    if dataset_subset == 'idle':
        data = data[data['state'] != 'operations']

    signals = np.stack(data.data.values)
    labels = data[target_label].to_numpy()

    # balance the dataset
    sampler = SMOTE()
    signals, labels = sampler.fit_resample(signals, labels)

    label_encoder = LabelEncoder()

    labels = label_encoder.fit_transform(labels)


    # Split the dataset
    val_size = int(len(signals) * val_split)
    train_size = len(signals) - val_size

    # apply standard scaer to the training data using the train_size
    scaler = StandardScaler()
    train_signals = scaler.fit_transform(signals[:train_size].reshape(-1, 1)).reshape(signals[:train_size].shape)
    # apply the standard scaler transform on the rest of the data using the train_size

    val_signals = scaler.transform(signals[train_size:].reshape(-1, 1)).reshape(signals[train_size:].shape)

    signals = np.concatenate([train_signals, val_signals]).reshape(signals.shape)


    # segment the dataset
    windows, window_labels = segment_dataset(signals, labels, sequence_length, overlap)
    # concat them to create the dataset
    signals = np.stack(windows)
    labels = np.stack(window_labels)

    return signals, labels, label_encoder.classes_



def encode_dataset_in_batches(model, dataset, num_workers=4, use_cuda=True):
    num_workers = cpu_count() // 2
    batch_size = 512
    print("num_workers: ", num_workers)
    print("batch_size: ", batch_size)
    model.eval()
    encoded_batches = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if use_cuda and torch.cuda.is_available():
        model = DataParallel(model).cuda()

    with torch.no_grad():
        # add tqdm here
        for batch in tqdm(dataloader):
            if use_cuda and torch.cuda.is_available():
                batch = batch.cuda(non_blocking=True)
            encoded = model.module.encoder(batch).detach()
            encoded_batches.append(encoded.cpu())  # Move to CPU here if necessary

    return torch.cat(encoded_batches).numpy()


def extract_features(model, data_module=None, train_dataloader=None, val_dataloader=None):
    """
    Extracts the segments and labels from the training and validation datasets
    """
    train_segments, train_labels = [], []
    test_segments, test_labels = [], []

    num_workers = cpu_count() // 2
    batch_size = 512
    print("num_workers: ", num_workers)
    print("batch_size: ", batch_size)

    if data_module:
        train_dataloader = data_module.train_dataloader(num_workers=num_workers, batch_size=batch_size)
        val_dataloader = data_module.val_dataloader(num_workers=num_workers, batch_size=batch_size)
    else:
        assert train_dataloader is not None and val_dataloader is not None, "Provide either a data module or the train and validation dataloaders"


    if torch.cuda.is_available():
        model = DataParallel(model).cuda()

    with torch.no_grad():
        for batch in tqdm(iter(train_dataloader)):
            segments, batch_labels = batch
            if torch.cuda.is_available():
                segments = segments.cuda(non_blocking=True)
                batch_labels = batch_labels.cuda(non_blocking=True)
            segments_encoded = model.module.encoder(segments).detach()
            train_segments.append(segments_encoded)
            train_labels.append(batch_labels)

        for batch in tqdm(iter(val_dataloader)):
            segments, batch_labels = batch
            if torch.cuda.is_available():
                segments = segments.cuda(non_blocking=True)
                batch_labels = batch_labels.cuda(non_blocking=True)
            segments_encoded = model.module.encoder(segments).detach()
            test_segments.append(segments_encoded)
            test_labels.append(batch_labels)

    # Flatten the segments and labels arrays
    X_train = torch.cat(train_segments).cpu().numpy()
    y_train = torch.cat(train_labels).cpu().numpy()

    X_test = torch.cat(test_segments).cpu().numpy()
    y_test = torch.cat(test_labels).cpu().numpy()

    return X_train, y_train, X_test, y_test


def extract_segments(data_module):
    """
    Extracts the segments and labels from the training and validation datasets
    """
    train_segments, train_labels = [], []
    test_segments, test_labels = [], []

    num_workers = cpu_count()
    batch_size = 512

    for batch in tqdm(data_module.train_dataloader(num_workers=num_workers, batch_size=batch_size)):
        segments, batch_labels = batch
        train_segments.append(segments)
        train_labels.append(batch_labels)

    for batch in tqdm(data_module.val_dataloader(num_workers=num_workers, batch_size=batch_size)):
        segments, batch_labels = batch
        test_segments.append(segments)
        test_labels.append(batch_labels)

    # Flatten the segments and labels arrays
    X_train = torch.cat(train_segments).numpy()
    y_train = torch.cat(train_labels).numpy()

    X_test = torch.cat(test_segments).numpy()
    y_test = torch.cat(test_labels).numpy()

    return X_train, y_train, X_test, y_test


class SegmentedSignalDataset(Dataset):
    def __init__(self, signals, labels, sequence_length, overlap, scaler=None,
                 **kwargs):
        """
        :param signals: A list or array of signals, each with the same length
        :param labels: A list or array of labels corresponding to each signal
        :param window_size: The size of the window to apply to each signal
        :param overlap_percent: The percentage of overlap between windows within the same signal
        """
        self.signals = signals
        self.labels = labels
        window_size = sequence_length
        self.window_size = window_size
        self.overlap_percent = overlap
        assert 0 <= overlap < 1, "Overlap percent must be between 0 and 1"


        self.label_encoder = LabelEncoder()

        self.step_size = window_size - int(window_size * self.overlap_percent)
        self.windows, self.window_labels = self._create_windows()

        # if scaler:
        #     self.windows = scaler.transform(self.windows.reshape(-1,1)).reshape(self.windows.shape)

    def _create_windows(self):
        windows = []
        window_labels = []
        for signal, label in zip(self.signals, self.labels):
            # Ensure each signal starts a new segmentation
            start = 0
            signal_length = len(signal)
            while start + self.window_size <= signal_length:
                end = start + self.window_size
                windows.append(signal[start:end])
                window_labels.append(label)
                start += self.step_size

        # windows = np.array(windows)
        # window_labels = np.array(window_labels)

        return windows, window_labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        label = self.window_labels[idx]
        window_tensor = torch.tensor(window, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)  # Assuming labels are for classification
        return window_tensor, label_tensor


class SegmentedSignalDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path,
                 sequence_length,
                 batch_size,
                 val_split,
                 dataset_subset,
                 target_label,
                 num_workers,
                 overlap,
                 use_class_weights=False,
                 **kwargs,
                 ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.val_split = val_split
        self.dataset = None
        self.target_names = None
        self.raw_signals = None
        self.raw_labels = None
        self.scaler = None
        self.label_encoder = None
        self.dataset_subset = dataset_subset
        self.overlap = overlap

        self.target_label = target_label

        self.class_weights = None
        self.use_class_weights = use_class_weights

        # take 90% of the cpus
        # self.num_workers = int(cpu_count() * 0.9)
        self.num_workers = num_workers
        # if self.trainer:
        #     self.num_workers = self.num_workers_per_cpu * self.trainer.num_gpus

    def setup(self, stage=None):
        # Set seed for reproducible splits
        # pl.seed_everything(42)
        data = load_data(self.dataset_path)

        data['class'] = 'normal'
        data['class'] = data['class'].mask(data['brand'] == 'OMG', 'anomaly')
        data['class'] = data['class'].mask(data['brand'] == 'teensyduino', 'anomaly')

        # make the length of the data equal to 100k
        data['data'] = data['data'].map(lambda x: x[:len(x)-4])

        if self.dataset_subset == 'idle':
            data = data[data['state'] != 'operations']

        elif self.dataset_subset == 'default':
            pass

        signals = np.stack(data.data.values)
        labels = data[self.target_label].to_numpy()

        # balance the dataset

        if not self.use_class_weights:
            smote = SMOTE()
            signals, labels = smote.fit_resample(signals, labels)

        # compute the class weights

        self.label_encoder = LabelEncoder()

        self.signals = signals
        self.labels = self.label_encoder.fit_transform(labels)

        if self.use_class_weights:
            self.class_weights = compute_class_weights(self.labels)

        # Split the dataset
        val_size = int(len(self.signals) * self.val_split)
        train_size = len(self.signals) - val_size

        # apply standard scaer to the training data using the train_size
        self.scaler = StandardScaler()
        train_signals = self.scaler.fit_transform(self.signals[:train_size].reshape(-1, 1)).reshape(self.signals[:train_size].shape)
        # apply the standard scaler transform on the rest of the data using the train_size

        val_signals = self.scaler.transform(self.signals[train_size:].reshape(-1, 1)).reshape(self.signals[train_size:].shape)

        self.signals = np.concatenate([train_signals, val_signals]).reshape(self.signals.shape)

        # Create the full dataset
        self.dataset = SegmentedSignalDataset(self.signals, self.labels, sequence_length=self.sequence_length, overlap=self.overlap)


        # compute the new train and val sizes
        val_size = int(len(self.dataset) * self.val_split)
        train_size = len(self.dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

        self.target_names = self.label_encoder.classes_


    def train_dataloader(self, num_workers=None, batch_size=None):
        if batch_size:
            self.batch_size = batch_size
        if num_workers:
            self.num_workers = num_workers
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self, num_workers=None, batch_size=None):
        if batch_size:
            self.batch_size = batch_size
        if num_workers:
            self.num_workers = num_workers
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=True)



class USBDataset(Dataset):
    """
    Dataset class for the USB dataset
    """
    def __init__(self, data, labels, window_size=config['sequence_length'], overlap=config['overlap']):
        """
        Args:
            data (ndarray): The data
            labels (ndarray): The labels
            window_size (int): The size of the sliding window
            overlap (float): The overlap between windows
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap))
        self.num_windows = (data.shape[1] - window_size) // self.stride + 1


        # Scale the data
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data.reshape(-1, 1)).reshape(self.data.shape)


        # Encode the labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return self.data.shape[0] * self.num_windows

    def __getitem__(self, idx):
        signal_idx = idx // self.num_windows
        window_idx = idx % self.num_windows
        start = window_idx * self.stride
        end = start + self.window_size
        return self.data[signal_idx, start:end], self.labels[signal_idx]


class USBDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path=config['dataset_path'], batch_size=config['batch_size'], val_split=config['val_split'], num_workers=4):
        super(USBDataModule, self).__init__()
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size
        self.val_split = val_split

        self.num_workers = num_workers * cpu_count()

    def setup(self, stage=None):
        data = load_data(self.dataset_path)

        data['class'] = 'normal'
        data['class'] = data['class'].mask(data['brand'] == 'OMG', 'anomaly')
        data['class'] = data['class'].mask(data['brand'] == 'teensyduino', 'anomaly')

        X = np.stack(data.data.values)
        y = data['category'].to_numpy()

        self.data = X
        self.labels = y
        dataset = USBDataset(self.data, self.labels)
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])

        self.target_names = dataset.label_encoder.classes_

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
