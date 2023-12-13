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


def save_dataset(dataset, dataset_name, target_label, config):

        # seed everything
        pl.seed_everything(config['seed'])

        # prepare the features and labels
        features = np.stack(dataset.data.values)
        labels = dataset[target_label].to_numpy()


        # encode the labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        target_names = label_encoder.classes_

        # segment the dataset to 10k length signals
        features, labels = segment_dataset(features, labels, window_size=10000, overlap=0.0)
        features = np.vstack(features)
        labels = np.vstack(labels).reshape(-1)

        num_classes = len(target_names)

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
    save_dataset(dataset_a, 'dataset_a', 'category', config)

    # Dataset B
    # characteristics:
    #   Identification task: individual devices from same category but different manufacturers
    #   (same type of category, but device with different serial number)
    # Choose a single device for each brand.
    print("Creating Dataset B")
    category = 'flash_drive'
    dataset_b = data[data['category'] == category]
    lexar = dataset_b[dataset_b['device'] == 'lexar_1']
    pny = dataset_b[dataset_b['device'] == 'pny_1']
    sandisk = dataset_b[dataset_b['device'] == 'sandisk_1']
    mosdat = dataset_b[dataset_b['device'] == 'mosdat']
    dataset_b = pd.concat([lexar, pny, sandisk, mosdat])
    save_dataset(dataset_b, 'dataset_b', 'device', config)

    # Dataset C1
    # characteristics:
    #   Identification task: individual devices same category and same manufacturer
    #   (same type of category, but device with different serial number))
    print("Creating Dataset C1")
    category = 'flash_drive'
    brand = 'pny'
    dataset_c1 = data[(data['category'] == category) & (data['brand'] == brand)]
    save_dataset(dataset_c1, 'dataset_c1', 'device', config)

    # Dataset C2
    # characteristics:
    #   Identification task: individual devices same category and same manufacturer
    #   (same type of category, but device with different serial number))
    print("Creating Dataset C2")
    category = 'keyboard'
    brand = 'dell'
    dataset_c2 = data[(data['category'] == category) & (data['brand'] == brand)]
    save_dataset(dataset_c2, 'dataset_c2', 'device', config)

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
    dataset_d1 = pd.concat([dell, perixx, lenovo, teensyduino])
    save_dataset(dataset_d1, 'dataset_d1', 'device', config)

    # Dataset D2
    # characteristics:
    #   Identification task: OMG behaving as cable against other cables
    #   Use 3-normal, remove WH and J-B.
    print("Creating Dataset D2")
    dataset_d2 = data[data['category'] == 'cable']
    dataset_d2 = dataset_d2[dataset_d2['device'] != 'WH']
    dataset_d2 = dataset_d2[dataset_d2['device'] != 'J-B']
    save_dataset(dataset_d2, 'dataset_d2', 'device', config)



if __name__ == '__main__':
    main()
