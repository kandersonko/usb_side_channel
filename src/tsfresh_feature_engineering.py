#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
import time

from utils.data import load_data, preprocess_features

import matplotlib.pyplot as plt

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



def feature_engineering(data_root_dir, target_label, subset, client, cluster):

    # load the dataset saved as numpy compressed file
    print("Loading the dataset")
    dataset_path = data_root_dir+f'{target_label}_dataset.npz'
    dataset = np.load(dataset_path, allow_pickle=True)

    X = dataset[f'X_{subset}']
    y = dataset[f'y_{subset}']

    print("Format the data for tsfresh")

    # data = format_data_for_tsfresh_dask(X, y, target_label, subset)
    ids = np.repeat(np.arange(X.shape[0]), X.shape[1])
    time_array = np.tile(np.arange(X.shape[1]), X.shape[0])
    df = pd.DataFrame({'id': ids, 'time': time_array, 'value': X.flatten()})

    # labels = np.repeat(y, X.shape[1])
    # df[target_label] = labels

    # calculate the number of partitions
    n_partitions = int(np.ceil(len(df) / 100000))
    print("n_partitions: ", n_partitions)

    data = dd.from_pandas(df, npartitions=n_partitions)

    print("Scatter the data to the workers")
    # scatter the data to the workers
    data = client.scatter(data, broadcast=True)

    # y = data[target_label].compute().values

    print(data.head())
    print(len(data))
    print(data.info())

    X_features = data.compute()
     # Repeat y labels for each time point in each series
    print("len X_features: ", len(X_features))
    print("len y: ", len(y))
    # print('y shape: ', y.shape)
    print("unique ids: ", len(X_features['id'].unique()), X_features['id'].unique())

    print(data.head())
    # print(client)

    # extract features using tsfresh with dask

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


    features = impute(features)

    # select relevant features
    # features.fillna(0, inplace=True)

    print("len features: ", len(features))
    print("len y: ", len(y))
    print("features head: ", features.head())
    # transform the labels to numpy arrays
    # features_filtered = select_features(features, y)
    # features_filtered = features_filtered.compute()

    end_time = time.time()
    duration = end_time - start_time

    plot_data.append({"name": "tsfresh", "task": "feature engineering", "dataset": f"{target_label}/{subset}", "duration": duration})
    # save the plot data
    plot_data = pd.DataFrame(plot_data)
    plot_data.to_csv(f'data/tsfresh_plot_data_{target_label}_{subset}.csv')

    features[target_label] = y
    features.to_csv(data_root_dir+f'{target_label}_{subset}_tsfresh.csv')

    # print("filtered features")
    # print(features_filtered.head())

    # save the features with the labels concatenated
    # features_filtered[target_label] = y
    # features_filtered.to_csv(data_root_dir+f'{target_label}_{subset}_tsfresh.csv')


def main():

    # create a dask cluster
    job_script_prologue = [
        "source ~/.bashrc",
        "conda activate usb",
        "unset LD_LIBRARY_PATH",
    ]
    cluster = SLURMCluster(
        cores=40,
        processes=8,
        memory="128GB",
        walltime="05:00:00",
        job_script_prologue=job_script_prologue,
        worker_extra_args=["--lifetime", "4h55m", "--lifetime-stagger", "4m"],
        # interface='ib0',
        death_timeout=120,
        # queue="reg",
        log_directory="jobs/dask_logs",
        # local_directory="/tmp/",
    )
    print(cluster)

    print(cluster.job_script())

    cluster.adapt(minimum=10, maximum=500)

    cluster.scale(100)

    # create a dask client
    client = Client(cluster)

    # extract features from the category dataset
    print("Extracting features from the category dataset")

    print("Extracting features from the train subset")
    feature_engineering(data_root_dir='datasets/',
                        target_label='category',
                        subset='train',
                        client=client,
                        cluster=cluster)

    print("Extracting features from the val subset")
    feature_engineering(data_root_dir='datasets/',
                        target_label='category',
                        subset='val',
                        client=client,
                        cluster=cluster)

    print("Extracting features from the test subset")
    feature_engineering(data_root_dir='datasets/',
                        target_label='category',
                        subset='test',
                        client=client,
                        cluster=cluster)

    # extract features from the class dataset
    print("Extracting features from the class dataset")

    print("Extracting features from the train subset")
    feature_engineering(data_root_dir='datasets/',
                        target_label='class',
                        subset='train',
                        client=client,
                        cluster=cluster)


    print("Extracting features from the val subset")
    feature_engineering(data_root_dir='datasets/',
                        target_label='class',
                        subset='val',
                        client=client,
                        cluster=cluster)

    print("Extracting features from the test subset")
    feature_engineering(data_root_dir='datasets/',
                        target_label='class',
                        subset='test',
                        client=client,
                        cluster=cluster)

    print("Done!")



if __name__ == '__main__':
    main()
