#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
import time

from utils.data import load_data, preprocess_features

import matplotlib.pyplot as plt

import datetime

#from imblearn.under_sampling import RandomUnderSampler

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.distribution import MultiprocessingDistributor
from tsfresh.feature_extraction import feature_calculators, settings

from tsfresh.utilities.distribution import ClusterDaskDistributor
from tsfresh.convenience.bindings import dask_feature_extraction_on_chunk

from tsfresh.utilities.dataframe_functions import impute

import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

import dask.dataframe as dd
from dask import optimize, compute

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report


def main(dataset_path):

    # dask.config.set({'distributed.scheduler.allowed-failures': 50})
    # dask.config.set({'distributed.scheduler.worker-ttl': None})

    # create a dask cluster
    cluster = SLURMCluster(cores=16,
                        processes=4,
                        memory="128GB",
                        walltime="02:00:00",
                        # interface='ib0',
                           #queue="reg",
                        )
    print(cluster)

    print(cluster.job_script())

    cluster.adapt(minimum=50, maximum=100)

    # create a dask client
    client = Client(cluster)

    cluster.scale(100)

    # load the dataset
    data_path = Path(dataset_path)
    data = dd.from_pandas(load_data(dataset_path), npartitions=10)

    # preprocess the features
    data['data'] = data['data'].map(lambda x: x[:len(x)-4])
    data['class'] = 'normal'
    data['class'] = data['class'].mask(data['brand'] == 'OMG', 'anomaly')
    data['class'] = data['class'].mask(data['brand'] == 'teensyduino', 'anomaly')
    data = data.persist()
    data['id'] = data.index
    y_category = data['category']
    y_category = y_category.compute()
    y_class = data['class']
    y_class = y_class.compute()
    data = data.explode('data')
    data['time'] = data.groupby('id').cumcount()
    data = data.rename(columns={'data': 'value'})

    print(client)
    print("y category shape: ", y_category.shape)
    print("y class shape: ", y_class.shape)
    print(data.head())

    data['value'] = data['value'].astype(np.float32)
    X = data[['id', 'time', 'class', 'value']].copy()
    X = X.rename(columns={"device": "kind"})
    # X = X.drop(columns=["class"])

    # extract features using tsfresh with dask

    # X_features = X.groupby(["id", "kind"])
    # X = X.compute()
    X_features = X.compute()

    plot_data = []

    start_time = time.time()

    distributor = ClusterDaskDistributor(cluster)

    print("Extracting features")

    features = extract_features(
        X_features,
        column_id="id",
        column_sort="time",
        column_value="value",
        distributor=distributor,
        default_fc_parameters=settings.EfficientFCParameters(),
        disable_progressbar=True,
    )

    # features = dask_feature_extraction_on_chunk(
    #     X,
    #     column_id="id",
    #     column_kind="kind",
    #     column_sort="time",
    #     column_value="value",
    #     disable_progressbar=True,
    # )

    impute(features)

    # select relevant features
    features.fillna(0, inplace=True)

    # transform the labels to numpy arrays
    features_filtered = select_features(features, y_category)
    # features_filtered = features_filtered.compute()

    end_time = time.time()
    duration = end_time - start_time

    plot_data.append({"name": "tsfresh", "task": "feature engineering", "dataset": "all", "duration": duration})
    # save the plot data
    plot_data = pd.DataFrame(plot_data)
    plot_data.to_csv('data/feature_engineering_plot_data.csv')

    print(features_filtered.head())

    # save the features with the labels concatenated
    features_filtered['class'] = y_class
    features_filtered['category'] = y_category
    features_filtered.to_csv('data/features_tsfresh.csv')


if __name__ == '__main__':
    main(dataset_path='data/datasets.pkl')
