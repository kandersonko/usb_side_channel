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



def extract_features(model, data_module):
    """
    Extracts encoded features and labels from the training and validation datasets
    """
    train_segments, train_category_labels, train_class_labels = [], [], []
    test_segments, test_category_labels, test_class_labels = [], [], []

    num_workers = cpu_count() // 2
    batch_size = 512
    print("num_workers: ", num_workers)
    print("batch_size: ", batch_size)

    if torch.cuda.is_available():
        model = DataParallel(model).cuda()

    with torch.no_grad():
        for batch in tqdm(data_module.train_dataloader(num_workers=num_workers, batch_size=batch_size)):
            segments, (category_labels, class_labels) = batch
            if torch.cuda.is_available():
                segments = segments.cuda(non_blocking=True)
                category_labels = category_labels.cuda(non_blocking=True)
                class_labels = class_labels.cuda(non_blocking=True)
            segments_encoded = model.module.encoder(segments).detach()
            train_segments.append(segments_encoded)
            train_category_labels.append(category_labels)
            train_class_labels.append(class_labels)

        for batch in tqdm(data_module.val_dataloader(num_workers=num_workers, batch_size=batch_size)):
            segments, (category_labels, class_labels) = batch
            if torch.cuda.is_available():
                segments = segments.cuda(non_blocking=True)
                category_labels = category_labels.cuda(non_blocking=True)
                class_labels = class_labels.cuda(non_blocking=True)
            segments_encoded = model.module.encoder(segments).detach()
            test_segments.append(segments_encoded)
            test_category_labels.append(category_labels)
            test_class_labels.append(class_labels)

    # Flatten the segments and labels arrays
    X_train = torch.cat(train_segments).cpu().numpy()
    y_train_category = torch.cat(train_category_labels).cpu().numpy()
    y_train_class = torch.cat(train_class_labels).cpu().numpy()

    X_test = torch.cat(test_segments).cpu().numpy()
    y_test_category = torch.cat(test_category_labels).cpu().numpy()
    y_test_class = torch.cat(test_class_labels).cpu().numpy()

    return X_train, y_train_category, y_train_class, X_test, y_test_category, y_test_class



def extract_segments(data_module):
    """
    Extracts the segments and labels from the training and validation datasets
    """
    train_segments, train_category_labels, train_class_labels = [], [], []
    test_segments, test_category_labels, test_class_labels = [], [], []

    num_workers = cpu_count()
    batch_size = 512

    # Extracting data from the training dataloader
    for batch in tqdm(data_module.train_dataloader(num_workers=num_workers, batch_size=batch_size)):
        segments, (category_labels, class_labels) = batch
        train_segments.append(segments)
        train_category_labels.append(category_labels)
        train_class_labels.append(class_labels)

    # Extracting data from the validation dataloader
    for batch in tqdm(data_module.val_dataloader(num_workers=num_workers, batch_size=batch_size)):
        segments, (category_labels, class_labels) = batch
        test_segments.append(segments)
        test_category_labels.append(category_labels)
        test_class_labels.append(class_labels)

    # Flatten the segments and labels arrays
    X_train = torch.cat(train_segments).numpy()
    y_train_category = torch.cat(train_category_labels).numpy()
    y_train_class = torch.cat(train_class_labels).numpy()

    X_test = torch.cat(test_segments).numpy()
    y_test_category = torch.cat(test_category_labels).numpy()
    y_test_class = torch.cat(test_class_labels).numpy()

    return X_train, y_train_category, y_train_class, X_test, y_test_category, y_test_class



class SegmentedSignalDataset(Dataset):
    def __init__(self, signals, labels, window_size=config['WINDOW_SIZE'], overlap_percent=config['OVERLAP'], scaler=None):
        """
        :param signals: A list or array of signals, each with the same length
        :param category_labels: A list or array of category labels corresponding to each signal
        :param class_labels: A list or array of class labels (normal/anomaly) corresponding to each signal
        :param window_size: The size of the window to apply to each signal
        :param overlap_percent: The percentage of overlap between windows within the same signal
        """
        self.signals = signals
        self.category_labels = category_labels
        self.class_labels = class_labels
        self.window_size = window_size
        self.overlap_percent = overlap_percent
        assert 0 <= overlap_percent < 1, "Overlap percent must be between 0 and 1"


        self.step_size = window_size - int(window_size * overlap_percent)
        self.windows, self.window_labels = self._create_windows()

        # if scaler:
        #     self.windows = scaler.transform(self.windows.reshape(-1,1)).reshape(self.windows.shape)

    def _create_windows(self):
        windows = []
        window_labels = []
        for signal, category_label, class_label in zip(self.signals, self.category_labels, self.class_labels):
            # Ensure each signal starts a new segmentation
            start = 0
            signal_length = len(signal)
            while start + self.window_size <= signal_length:
                end = start + self.window_size
                windows.append(signal[start:end])
                window_labels.append((category_label, class_label))
                start += self.step_size

        return windows, window_labels


    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        category_label, class_label = self.window_labels[idx]
        window_tensor = torch.tensor(window, dtype=torch.float32)
        category_label_tensor = torch.tensor(category_label, dtype=torch.long)
        class_label_tensor = torch.tensor(class_label, dtype=torch.long)
        return window_tensor, (category_label_tensor, class_label_tensor)


class SegmentedSignalDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path=config['DATASET_PATH'],
                 batch_size=config['BATCH_SIZE'],
                 val_split=config['VAL_SPLIT'],
                 dataset_subset=config['DATASET_SUBSET'],
                 ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size
        self.val_split = val_split
        self.dataset = None
        self.target_names = None
        self.raw_signals = None
        self.raw_labels = None
        self.scaler = None
        self.label_category_encoder = None
        self.dataset_subset = dataset_subset

        # take 90% of the cpus
        # self.num_workers = int(cpu_count() * 0.9)
        self.num_workers = cpu_count() // 2
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

        # Prepare signals
        signals = np.stack(data.data.values)
        category_labels = data['category'].to_numpy()
        class_labels = data['class'].to_numpy()

        # Encoding category labels
        category_label_encoder = LabelEncoder()
        encoded_category_labels = category_label_encoder.fit_transform(category_labels)

        # Encoding class labels (anomaly detection)
        class_label_encoder = LabelBinarizer()
        encoded_class_labels = class_label_encoder.fit_transform(class_labels)

        # Combining category and class labels for SMOTE
        combined_labels = np.column_stack((encoded_category_labels, encoded_class_labels))

        # Balancing the dataset with SMOTE
        smote = SMOTE()
        signals, combined_labels = smote.fit_resample(signals, combined_labels)

        # Separating the combined labels
        self.category_labels = combined_labels[:, :-1].flatten()
        self.class_labels = combined_labels[:, -1]

        # Split the dataset
        val_size = int(len(signals) * self.val_split)
        train_size = len(signals) - val_size

        # Scaling the data
        self.scaler = StandardScaler()
        train_signals = self.scaler.fit_transform(signals[:train_size])
        val_signals = self.scaler.transform(signals[train_size:])

        # Preparing the final dataset
        self.train_dataset = SegmentedSignalDataset(train_signals, self.category_labels[:train_size], self.class_labels[:train_size])
        self.val_dataset = SegmentedSignalDataset(val_signals, self.category_labels[train_size:], self.class_labels[train_size:])

        # Saving the label names
        self.category_target_names = category_label_encoder.classes_
        self.class_target_names = class_label_encoder.classes_

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
    def __init__(self, data, labels, window_size=config['WINDOW_SIZE'], overlap=config['OVERLAP']):
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
    def __init__(self, dataset_path=config['DATASET_PATH'], batch_size=config['BATCH_SIZE'], val_split=config['VAL_SPLIT'], num_workers=4):
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
