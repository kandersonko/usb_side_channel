#!/usr/bin/env python3

# environment variables to prevent over provisioning
import os
os.environ['OMP_NUM_THREADS'] = "1"
os.environ['MKL_NUM_THREADS'] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

import time
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

from tsfresh.utilities.dataframe_functions import impute, impute_dataframe_zero

import dask
from dask.distributed import Client, LocalCluster
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
# import cpu_count
from multiprocessing import cpu_count

from config import default_config, merge_config_with_cli_args



def process_chunk(chunk_x, chunk_id):
    """
    process a single chunk of data.

    args:
        chunk_x: the chunk of the dataset to be processed.
        chunk_id: identifier for the chunk.

    returns:
        dataframe containing features extracted from the chunk.
    """
    ids = np.repeat(np.arange(chunk_x.shape[0]), chunk_x.shape[1])
    time_array = np.tile(np.arange(chunk_x.shape[1]), chunk_x.shape[0])
    df_chunk = pd.DataFrame({'id': ids, 'time': time_array, 'value': chunk_x.flatten()})

    # convert chunk to dask dataframe
    n_cores = int(os.getenv("NUMBER_OF_CPUS") or cpu_count())
    n_jobs = max(1, n_cores // 2)
    data_chunk = dd.from_pandas(df_chunk, npartitions=10)

    # extract features using tsfresh with dask for the chunk

    # distributor = ClusterDaskDistributor(cluster)

    # compute the number of n_jobs based on the number of workers
    #
    default_fc_parameters = settings.EfficientFCParameters()

    chunk_features = extract_features(
        data_chunk,
        column_id="id",
        column_sort="time",
        column_value="value",
        # distributor=distributor,
        default_fc_parameters=default_fc_parameters,
        disable_progressbar=True,
        # pivot=False,
        n_jobs=n_jobs, # default to all cores
        # impute_function=impute,
    )


    return chunk_features


def feature_engineering(data_root_dir, target_label, subset, client, cluster, chunk_size):

    # load the dataset saved as numpy compressed file
    print("Loading the dataset")
    dataset_path = data_root_dir+f'{target_label}_dataset.npz'
    dataset = np.load(dataset_path, allow_pickle=True)

    X = dataset[f'X_{subset}']
    # y = dataset[f'y_{subset}']

    ids = np.repeat(np.arange(X.shape[0]), X.shape[1])
    time_array = np.tile(np.arange(X.shape[1]), X.shape[0])
    df = pd.DataFrame({'id': ids, 'time': time_array, 'value': X.flatten()})

    data = dd.from_pandas(df, npartitions=chunk_size)

    data = client.persist(data)

    # convert chunk to dask dataframe
    n_cores = int(os.getenv("NUMBER_OF_CPUS") or cpu_count())
    n_jobs = max(1, n_cores // 2)

    # extract features using tsfresh with dask for the chunk

    # distributor = ClusterDaskDistributor(cluster)

    # compute the number of n_jobs based on the number of workers
    #



    print("X shape: ", X.shape)
    # print("y shape: ", y.shape)


    plot_data = []

    # distributor = ClusterDaskDistributor(cluster)

    print("Extracting features")

    start_time = time.time()

    # chunk_features_list = []

    # print("Processing the chunks")
    # # Process each chunk
    # for chunk_start in tqdm(range(0, len(X), chunk_size)):
    #     chunk_X = X[chunk_start:chunk_start + chunk_size]

    #     # default_fc_parameters = settings.EfficientFCParameters()
    #     chunk_features = process_chunk(chunk_X, cluster, client, chunk_start, default_fc_parameters)
    #     # chunk_features = dask.delayed(process_chunk)(chunk_X, chunk_start)
    # print("Computing the chunks")
    # # Gather the results from the workers
    # chunk_features_list = dask.compute(*chunk_features_list)
    # # Combine features from all chunks
    # features = pd.concat(chunk_features_list, axis=0)

    partitions = data.to_delayed()
    default_fc_parameters = settings.EfficientFCParameters()

    chunk_features = [dask.delayed(extract_features)(
        data_chunk,
        column_id="id",
        column_sort="time",
        column_value="value",
        # distributor=distributor,
        default_fc_parameters=default_fc_parameters,
        disable_progressbar=True,
        # pivot=False,
        n_jobs=1, # default to all cores
        # impute_function=impute,
    ) for data_chunk in partitions]

    chunk_features = dask.compute(*chunk_features)

    features = pd.concat(chunk_features, axis=0)

    print("features: ", features.head())


    print("imputing features")
    features = impute(features)

    # print("selecting features")
    # features = select_features(features, y)

    end_time = time.time()
    duration = end_time - start_time

    print("Saving the plot data")

    plot_data.append({"name": "tsfresh", "task": "feature engineering", "dataset": f"{target_label}/{subset}", "duration": duration})
    # save the plot data
    plot_data = pd.DataFrame(plot_data)
    plot_data.to_csv(f'data/tsfresh_plot_data_{target_label}_{subset}.csv')

    # save the features with the labels
    print("Saving the features")
    # features[target_label] = y
    features.to_csv(data_root_dir+f'{target_label}_{subset}_tsfresh.csv')



def main():
    config = merge_config_with_cli_args(default_config)

    use_local_cluster = config.get('use_local_cluster', True)

    n_cores = int(os.getenv("NUMBER_OF_CPUS") or cpu_count())

    client = None
    cluster = None

    n_workers = max(config.get('workers', 1), n_cores//2)
    memory = config.get('memory', '2GB')

    print(f"n_workers: {n_workers}, memory: {memory}, n_cores: {n_cores}")

    if use_local_cluster:

        # cluster = LocalCluster(
        #     n_workers=n_workers,
        #     memory_limit=memory,
        # )
        # client = Client(cluster)
        client = Client(memory_limit=memory)
        cluster = client.cluster

        cluster.adapt(minimum=n_workers, maximum=n_cores)

    else:
        # create a dask cluster
        job_script_prologue = [
            "source ~/.bashrc",
            "conda activate usb",
            # "unset LD_LIBRARY_PATH",
        ]
        cluster = SLURMCluster(
            cores=24,
            processes=4,
            memory="16GB",
            walltime="01:00:00",
            shebang="#!/usr/bin/env sh",
            job_script_prologue=job_script_prologue,
            worker_extra_args=["--lifetime", "55m", "--lifetime-stagger", "4m"],
            # interface='ib0',
            queue="gpu-8",
            # death_timeout=120,
            log_directory="jobs/dask_logs",
            # local_directory="/tmp/",
        )
        print(cluster.job_script())
        cluster.adapt(minimum=10, maximum=500)

        # create a dask client
        client = Client(cluster)


    print("client: ", client)
    print("cluster: ", cluster)

    # scale the dask workers
    cluster.scale(n_cores)

    print("Waiting for workers to be ready")
    time.sleep(5)
    print("Workers ready: ", len(client.scheduler_info().get("workers")))

    target_label = config.get('target_label')

    subset = config.get('dataset_subset')

    chunk_size = config.get('chunk_size', 10)

    print(f"Extracting features from the {target_label} dataset")

    print(f"Extracting features from the {subset} subset")
    feature_engineering(
        data_root_dir='datasets/',
        target_label=target_label,
        subset=subset,
        client=client,
        cluster=cluster,
        chunk_size=chunk_size,
    )


    # close the dask client
    client.close()
    cluster.close()

    print("Done!")

if __name__ == '__main__':
    main()
