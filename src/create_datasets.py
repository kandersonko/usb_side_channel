#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import lightning.pytorch as pl
from config import default_config, merge_config_with_cli_args
from utils.data import load_data
from dataset import segment_dataset


def save_dataset(dataset, dataset_name, target_label, config, common_data):

        # seed everything
        pl.seed_everything(config['seed'])

        # prepare the features and labels
        features = np.stack(dataset.data.values)
        labels = dataset[target_label].to_numpy()
        # print("Dataset shape:", features.shape, labels.shape)

        # encode the labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        target_names = label_encoder.classes_

        # Initialize lists to hold the sampled features and labels
        sampled_features = []
        sampled_labels = []

        # Iterate over each class and sample 1000 features
        for class_idx in np.unique(labels):
                class_features = features[labels == class_idx]
                class_labels = labels[labels == class_idx]

                # Randomly sample 1000 instances or all if less than 1000
                n_samples = min(config['max_samples_per_class'], class_features.shape[0])
                sampled_idx = np.random.choice(class_features.shape[0], n_samples, replace=False)

                sampled_features.append(class_features[sampled_idx])
                sampled_labels.append(class_labels[sampled_idx])

        # Combine the sampled features and labels
        features = np.vstack(sampled_features)
        labels = np.concatenate(sampled_labels)

        print("Dataset shape:", features.shape, labels.shape)

        print("Segmenting the dataset")
        # segment the dataset to 10k length signals
        features, labels = segment_dataset(features, labels, window_size=config['sequence_length'], overlap=config['overlap'])
        features = np.vstack(features)
        labels = np.vstack(labels).reshape(-1)

        print("Dataset shape:", features.shape, labels.shape)
        print(np.unique(labels, return_counts=True))

        num_classes = len(target_names)

        print("Number of classes:", num_classes)
        print("Number of unique labels:", len(np.unique(labels)))

        # split the dataset into train, val and test
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=config['val_split'], random_state=config['seed'],
            stratify=labels,
        )

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=config['val_split'], random_state=config['seed'],
            stratify=y_train,
        )

        common_data['X_train'].append(X_train)
        common_data['X_val'].append(X_val)

        # save the dataset to disk
        np.savez_compressed(
            os.path.join(config['data_dir'], f'{dataset_name}-{target_label}.npz'),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            target_names=target_names,
            num_classes=num_classes,
        )

def main():

    config = merge_config_with_cli_args(default_config)

    # seed everything
    seed = config['seed']
    pl.seed_everything(seed)

    dataset_path = config['dataset_path']
    # load the dataset
    data = load_data(dataset_path)
    # make the length of the signals 100k
    data['data'] = data['data'].map(lambda x: x[:len(x)-4])
    # create a new column using the device category and name
    data['device_name'] = data['category'] + '/' + data['device']
    # rename captures state to idle
    data['state'] = data['state'].map(lambda x: 'idle' if x == 'captures' else x)

    # select only idle state
    data = data[data['state'] == 'idle']

    common_data = dict(X_train=[], X_val=[])

    # Dataset A
    # characteristics:
    #   Identification task: device categories (keyboard, mouse, etc.),
    #   select one brand, and one device from each category
    # select one brand, and one device from that brand
    print("Creating Dataset A")
    brand = 'perixx'
    keyboard_and_mouse = data[data['brand'] == brand]
    flash_drive = data[data['device'] == 'sandisk_1']
    cable = data[data['device'] == 'HD']
    dataset_a = pd.concat([keyboard_and_mouse, flash_drive, cable])
    save_dataset(dataset_a, 'dataset_a', 'category', config, common_data)

    # Dataset B
    # characteristics:
    #   Identification task: individual devices from same category but different manufacturers
    #   (same type of category, but device with different serial number)
    # Choose a single device for each brand.
    print("Creating Dataset B")
    # category = 'flash_drive'
    category = 'mouse'
    dataset_b = data[data['category'] == category]
    # flash drive devices
    # lexar = dataset_b[dataset_b['device'] == 'lexar_1']
    # pny = dataset_b[dataset_b['device'] == 'pny_1']
    # sandisk = dataset_b[dataset_b['device'] == 'sandisk_1']
    # # sandisk = dataset_b[dataset_b['device'] == 'sandisk_glide']
    # mosdat = dataset_b[dataset_b['device'] == 'mosdat']
    # dataset_b = pd.concat([lexar, pny, sandisk, mosdat])
    #
    # mouse devices
    logitech = dataset_b[dataset_b['device'] == 'logitech_1']
    perixx = dataset_b[dataset_b['device'] == 'perixx']
    dell = dataset_b[dataset_b['device'] == 'dell']
    lenovo = dataset_b[dataset_b['device'] == 'lenovo']
    dataset_b = pd.concat([logitech, perixx, dell, lenovo])
    save_dataset(dataset_b, 'dataset_b', 'device', config, common_data)

    # Dataset C1
    # characteristics:
    #   Identification task: individual devices same category and same manufacturer
    #   (same type of category, but device with different serial number))
    print("Creating Dataset C1")
    # flash drive devices
    # category = 'flash_drive'
    # brand = 'pny'
    #
    # mouse devices
    category = 'mouse'
    brand = 'logitech'
    dataset_c1 = data[(data['category'] == category) & (data['brand'] == brand)]
    save_dataset(dataset_c1, 'dataset_c1', 'device', config, common_data)

    # Dataset C2
    # characteristics:
    #   Identification task: individual devices same category and same manufacturer
    #   (same type of category, but device with different serial number))
    print("Creating Dataset C2")
    category = 'keyboard'
    brand = 'dell'
    dataset_c2 = data[(data['category'] == category) & (data['brand'] == brand)]
    save_dataset(dataset_c2, 'dataset_c2', 'device', config, common_data)

    # Dataset D1
    # characteristics:
    #   Identification task: microcontroller behaving as keyboard against other keyboards
    #   Choose 1 device for each brand (choose 1 each, dell→1, lenovo→1,etc.)
    print("Creating Dataset D1")
    dataset_d1 = data[(data['category'] == 'keyboard') | (data['category'] == 'microcontroller')]
    dell = dataset_d1[dataset_d1['device'] == 'dell_1']
    perixx = dataset_d1[dataset_d1['device'] == 'perixx']
    lenovo = dataset_d1[dataset_d1['device'] == 'lenovo']
    teensyduino = dataset_d1[dataset_d1['device'] == 'teensyduino']
    # take 10 samples from teensyduino
    teensyduino = teensyduino.sample(n=10, random_state=seed)
    dataset_d1 = pd.concat([dell, perixx, lenovo, teensyduino])
    save_dataset(dataset_d1, 'dataset_d1', 'device', config, common_data)

    # Dataset D2
    # characteristics:
    #   Identification task: OMG behaving as cable against other cables
    #   Use 3-normal, remove WH and J-B.
    print("Creating Dataset D2")
    dataset_d2 = data[data['category'] == 'cable']
    # dataset_d2 = dataset_d2[dataset_d2['device'] != 'WH']
    dataset_d2 = dataset_d2[dataset_d2['device'] != 'NT']
    dataset_d2 = dataset_d2[dataset_d2['device'] != 'J-B']
    save_dataset(dataset_d2, 'dataset_d2', 'device', config, common_data)

    # save the common data
    np.savez_compressed(
        os.path.join(config['data_dir'], f'common_dataset.npz'),
        X_train=np.vstack(common_data['X_train']),
        X_val=np.vstack(common_data['X_val']),
    )


if __name__ == '__main__':
    main()
