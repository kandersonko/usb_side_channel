#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
import time
from tqdm import tqdm

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


def process_chunk(chunk_X, cluster, client, chunk_id, default_fc_parameters):
    """
    Process a single chunk of data.

    Args:
        chunk_X: The chunk of the dataset to be processed.
        cluster: The Dask cluster.
        chunk_id: Identifier for the chunk.
        default_fc_parameters: The feature extraction settings for tsfresh.

    Returns:
        DataFrame containing features extracted from the chunk.
    """
    print(f"Processing chunk {chunk_id}")
    # gather the chunk to the workers
    chunk_X = client.gather(chunk_X)

    ids = np.repeat(np.arange(chunk_X.shape[0]), chunk_X.shape[1])
    time_array = np.tile(np.arange(chunk_X.shape[1]), chunk_X.shape[0])
    df_chunk = pd.DataFrame({'id': ids, 'time': time_array, 'value': chunk_X.flatten()})

    # Convert chunk to Dask DataFrame
    data_chunk = dd.from_pandas(df_chunk, npartitions=10)  # Adjust npartitions as needed

    # Extract features using tsfresh with Dask for the chunk
    distributor = ClusterDaskDistributor(cluster)
    chunk_features = extract_features(
        data_chunk.compute(),
        column_id="id",
        column_sort="time",
        column_value="value",
        distributor=distributor,
        default_fc_parameters=default_fc_parameters,
        disable_progressbar=True,
    )
    # features = dask_feature_extraction_on_chunk(
    #     df_grouped,
    #     column_id="id",
    #     column_kind=target_label,
    #     column_sort="time",
    #     column_value="value",
    #     default_fc_parameters=settings.EfficientFCParameters(),
    # )

    return chunk_features


def feature_engineering(data_root_dir, target_label, subset, client, cluster):

    # load the dataset saved as numpy compressed file
    print("Loading the dataset")
    dataset_path = data_root_dir+f'{target_label}_dataset.npz'
    dataset = np.load(dataset_path, allow_pickle=True)

    X = dataset[f'X_{subset}']
    y = dataset[f'y_{subset}']

    print("X shape: ", X.shape)
    print("y shape: ", y.shape)

    print("Format the data for tsfresh")

    plot_data = []


    distributor = ClusterDaskDistributor(cluster)


    print("Extracting features")

    # Determine chunk size
    chunk_size = 10000

    # Initialize empty list to collect features from each chunk
    chunk_features_list = []

    start_time = time.time()

    # Process each chunk
    for chunk_start in range(0, len(X), chunk_size):
        chunk_X = X[chunk_start:chunk_start + chunk_size]
        # scatter the chunk to the workers
        chunk_X = client.scatter(chunk_X, broadcast=True)
        chunk_features = process_chunk(chunk_X, cluster, client, chunk_start, settings.EfficientFCParameters())
        chunk_features_list.append(chunk_features)

    # Combine features from all chunks
    combined_features = pd.concat(chunk_features_list)

    features = impute(combined_features)

    # select relevant features

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
        cores=16,
        processes=4,
        memory="16GB",
        walltime="01:00:00",
        job_script_prologue=job_script_prologue,
        worker_extra_args=["--lifetime", "55m", "--lifetime-stagger", "4m"],
        # interface='ib0',
        death_timeout=120,
        # queue="reg",
        log_directory="jobs/dask_logs",
        # local_directory="/tmp/",
    )
    print(cluster)

    print(cluster.job_script())

    cluster.adapt(minimum=10, maximum=500)

    cluster.scale(10)

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
