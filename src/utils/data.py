import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# import tensorflow as tf


# def create_dataset(X, Y, batch_size):
#     """ 
#     Create train and test TF dataset from X and Y
#     The prefetch overlays the preprocessing and model execution of a training step.
#     While the model is executing training step s, the input pipeline is reading the data for step s+1.
#     AUTOTUNE automatically tune the number for sample which are prefeteched automatically.

#     Keyword arguments:
#     X -- numpy array
#     Y -- numpy array
#     batch_size -- integer
#     """
#     # from # https://gist.github.com/rameshKrSah/c6dea6fada460be48499e2ff4399562

#     AUTOTUNE = tf.data.experimental.AUTOTUNE

#     X = X.astype('float32')
#     Y = Y.astype('float32')

#     x_tr, x_ts, y_tr, y_ts = train_test_split(
#         X, Y, test_size=0.2, random_state=42, stratify=Y, shuffle=True)

#     train_dataset = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
#     train_dataset = train_dataset.shuffle(
#         buffer_size=1000, reshuffle_each_iteration=True)
#     train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)

#     test_dataset = tf.data.Dataset.from_tensor_slices((x_ts, y_ts))
#     test_dataset = test_dataset.batch(batch_size).prefetch(AUTOTUNE)

#     return train_dataset, test_dataset


def preprocess_features(features, labels, window_len=500, max_features_len=100000):
    features_ = []
    labels_ = []
    for (label, feature) in zip(labels, features):
        reshaped_feature = np.reshape(
            feature[:max_features_len], (-1, window_len,))
        features_.append(reshaped_feature)
        labels_.append([label]*reshaped_feature.shape[0])
    features_ = np.concatenate(np.array(features_), axis=0)
    labels_ = np.concatenate(np.array(labels_), axis=0)
    return features_, labels_


def prepare_labels(labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    classes = {}
    target_classes = {}
    for label in label_encoder.classes_:
        val = label_encoder.transform([label])[0]
        classes[val] = label
        target_classes[label] = val
    return classes, target_classes, label_encoder


def normalize(features, labels, label_encoder):
    # normalization
    std = StandardScaler()

    features_ = std.fit_transform(features)
    labels_ = label_encoder.transform(labels)

    X, y = features_, labels_

    return X, y


def preprocess(dataset, features_column_name, label_column_name, window_len=500):
    # Preprocess features
    features, labels = preprocess_features(
        dataset[features_column_name], dataset[label_column_name], window_len=window_len)
    classes, target_classes, label_encoder = prepare_labels(labels)
    # normalization
    features, labels = normalize(features, labels, label_encoder)
    return features, labels, label_encoder


def load_data(data_path):

    df = pd.read_pickle(data_path)

    # microcontroller
    # teensy = df[df['brand'] == 'teensyduino']
    # microcontroller = teensy
    df.loc[df['brand'] == 'teensyduino', 'category'] = 'microcontroller'
    microcontroller = df[df['category'] == 'microcontroller']

    # cables = df[df['category'] == 'cables']
    df.loc[df['category'] == 'cables', 'category'] = 'cable'
    cables = df[df['category'] == 'cable']

    # keyboards = df[df['category'] == 'keyboards']
    df.loc[df['category'] == 'keyboards', 'category'] = 'keyboard'
    keyboards = df[df['category'] == 'keyboard']

    # mice = df[df['category'] == 'mice']
    df.loc[df['category'] == 'mice', 'category'] = 'mouse'
    mice = df[df['category'] == 'mouse']

    # flash_drives = df[df['category'] == 'flash_drives']
    df.loc[df['category'] == 'flash_drives', 'category'] = 'flash_drive'
    flash_drives = df[df['category'] == 'flash_drive']

    # Balancing the dataset

    # smallest_length = min(len(cables), len(keyboards), len(mice), len(microcontroller),
    #                       len(flash_drives))

    data_items = [cables, keyboards, mice, microcontroller, flash_drives]
    # min_length = np.Infinity
    # for item in data_items:
    #     min_length = min(min_length, len(item))

    # data = []
    # for item in data_items:
    #     data.append(item.sample(min_length))

    # dataset = pd.concat(data)

    dataset = pd.concat(data_items)

    return dataset


def prepare_dataset(dataset_path):
    data = load_data(dataset_path)
    data['class'] = 'normal'
    # add anomaly column for OMG and teensyduino
    data.loc[data['brand'] == 'OMG', 'class'] = 'anomaly'
    data.loc[data['brand'] == 'teensyduino', 'class'] = 'anomaly'
    return data
