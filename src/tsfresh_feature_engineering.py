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
from dask import optimize
from dask import delayed, compute

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# environment variables to prevent over provisioning
import os
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"


def process_chunk(chunk_x, cluster, client, chunk_id, default_fc_parameters):
    """
    process a single chunk of data.

    args:
        chunk_x: the chunk of the dataset to be processed.
        cluster: the dask cluster.
        chunk_id: identifier for the chunk.
        default_fc_parameters: the feature extraction settings for tsfresh.

    returns:
        dataframe containing features extracted from the chunk.
    """
    print(f"processing chunk {chunk_id}")
    # gather the chunk to the workers
    # chunk_x = client.gather(chunk_x)

    ids = np.repeat(np.arange(chunk_x.shape[0]), chunk_x.shape[1])
    time_array = np.tile(np.arange(chunk_x.shape[1]), chunk_x.shape[0])
    df_chunk = pd.DataFrame({'id': ids, 'time': time_array, 'value': chunk_x.flatten()})

    # convert chunk to dask dataframe
    data_chunk = dd.from_pandas(df_chunk, npartitions=5)  # adjust npartitions as needed

    # extract features using tsfresh with dask for the chunk

    distributor = ClusterDaskDistributor(cluster)

    chunk_features = extract_features(
        data_chunk.compute(),
        column_id="id",
        column_sort="time",
        column_value="value",
        distributor=distributor,
        default_fc_parameters=default_fc_parameters,
        disable_progressbar=True,
        pivot=False,
        n_jobs=4,
    )

    # df_grouped = data_chunk.groupby(["id", "kind"])
    # chunk_features = dask_feature_extraction_on_chunk(
    #     df_grouped,
    #     column_id="id",
    #     column_kind=target_label,
    #     column_sort="time",
    #     column_value="value",
    #     default_fc_parameters=settings.efficientfcparameters(),
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


    # distributor = ClusterDaskDistributor(cluster)


    print("Extracting features")

    # Determine chunk size
    chunk_size = 1000

    # Initialize empty list to collect features from each chunk
    chunk_features_list = []

    start_time = time.time()

    # Process each chunk
    for chunk_start in tqdm(range(0, len(X), chunk_size)):
        chunk_X = X[chunk_start:chunk_start + chunk_size]
        # scatter the chunk to the workers
        # chunk_X = client.scatter(chunk_X, broadcast=True)
        chunk_features = process_chunk(chunk_X, cluster, client, chunk_start, settings.EfficientFCParameters())
        chunk_features_list.append(chunk_features)

    # Compute all tasks in parallel
    chunk_features_list = compute(*delayed_tasks)

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
        memory="8GB",
        walltime="00:30:00",
        shebang="#!/usr/bin/env sh",
        job_script_prologue=job_script_prologue,
        worker_extra_args=["--lifetime", "25m", "--lifetime-stagger", "4m"],
        # interface='ib0',
        queue="short",
        # death_timeout=120,
        log_directory="jobs/dask_logs",
        # local_directory="/tmp/",
    )
    print(cluster)

    print(cluster.job_script())

    cluster.adapt(minimum=10, maximum=500)

    cluster.scale(20)

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
