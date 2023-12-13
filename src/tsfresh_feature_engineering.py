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
# from tsfresh.utilities.distribution import MultiprocessingDistributor
from tsfresh.feature_extraction import feature_calculators, settings
from tsfresh.feature_extraction.data import to_tsdata

from tsfresh.utilities.distribution import ClusterDaskDistributor, LocalDaskDistributor
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


def create_df(sample_id, data):
    return pd.DataFrame({
        'id': sample_id,
        'time': range(data.shape[1]),
        'value': data[sample_id, :]
    })

def process_chunk(chunk_x, chunk_id):
    """
    process a single chunk of data.

    args:
        chunk_x: the chunk of the dataset to be processed.
        chunk_id: identifier for the chunk.

    returns:
        dataframe containing features extracted from the chunk.
    """

    dfs = []
    for sample_id in range(chunk_x.shape[0]):
        df_sample = create_df(sample_id, chunk_x)
        dfs.append(df_sample)

    df_chunk = pd.concat(dfs)

    data_chunk = dd.from_pandas(df_chunk, npartitions=10)

    # convert chunk to dask dataframe
    n_cores = int(os.getenv("NUMBER_OF_CPUS") or cpu_count())
    n_jobs = max(1, n_cores // 2)

    # extract features using tsfresh with dask for the chunk


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
        n_jobs=1, # default to all cores
        # impute_function=impute,
    )


    return chunk_features


def feature_engineering(data_root_dir, target_label, subset, client, cluster, distributor, chunk_size):

    global_start_time = time.time()
    # load the dataset saved as numpy compressed file
    print("Loading the dataset")
    dataset_path = data_root_dir+f'/{target_label}_dataset.npz'
    dataset = np.load(dataset_path, allow_pickle=True)

    X = dataset[f'X_{subset}']
    # y = dataset[f'y_{subset}']

    print("X shape: ", X.shape)
    # print("y shape: ", y.shape)

    # Populate the DataFrame
    dfs = []
    for sample_id in range(X.shape[0]):
        df_sample = create_df(sample_id, X)
        dfs.append(df_sample)

    data = pd.concat(dfs)


    # data = dd.from_pandas(df, npartitions=chunk_size)

    # data = client.persist(data)

    # data = data.compute()

    print("data: ", data.head())
    print("len(data): ",  len(data))

    plot_data = []


    print("Extracting features")

    start_time = time.time()

    default_fc_parameters = settings.EfficientFCParameters()

    n_workers = len(client.scheduler_info().get("workers"))

    # data = dd.from_pandas(data, chunksize=chunk_size)


    # data = dd.from_pandas(data, npartitions=chunk_size)

    # chunk_size, extra = divmod(len(X), n_workers * 5)
    # if extra:
    #     chunk_size += 1

    # partitions = data.to_delayed()

    # client.scatter(partitions)

    # data = client.persist(data)

    # chunk_features = [dask.delayed(extract_features)(
    #     data_chunk,
    #     column_id="id",
    #     column_sort="time",
    #     column_value="value",
    #     # distributor=distributor,
    #     default_fc_parameters=default_fc_parameters,
    #     disable_progressbar=True,
    #     pivot=False,
    #     n_jobs=0, # default to all cores
    #     # chunksize=chunk_size,
    #     # impute_function=impute,
    # ) for data_chunk in partitions]

    # chunk_features = dask.compute(*chunk_features)

    # features = pd.concat(chunk_features, axis=0)


    features = extract_features(
        data,
        column_id="id",
        column_sort="time",
        column_value="value",
        distributor=distributor,
        default_fc_parameters=default_fc_parameters,
        disable_progressbar=True,
        pivot=False,
        # n_jobs=n_workers, # default to all cores
        # chunksize=chunk_size,
        # impute_function=impute,
    )

    # pivot the data
    data_features = to_tsdata(data, column_id="id", column_sort="time", column_value="value")

    features = data_features.pivot(features)

    print("features: ")
    print(features.head())
    # print("features[0]: ", type(features[0]))

    # features = features.compute()

    print("Imputing features")
    features = impute(features)


    print("features: ", features.head())
    print("features length: ", len(features))
    print("X shape: ", X.shape)


    # print("imputing features")
    # features = dd.from_pandas(features, npartitions=chunk_size)
    # features = dask.delayed(impute)(features)
    # features = dask.compute(features)

    end_time = time.time()
    duration = end_time - start_time

    print("Saving the plot data")

    plot_data.append({"name": "tsfresh", "task": "feature engineering", "dataset": f"{target_label}/{subset}", "duration": duration})
    # save the plot data
    plot_data = pd.DataFrame(plot_data)
    plot_data.to_csv(f'data/tsfresh_plot_data_{target_label}_{subset}.csv')

    # save the features with the labels
    print("Saving the features")
    features.to_csv(data_root_dir+f'/{target_label}_{subset}_tsfresh.csv')

    global_end_time = time.time()
    global_duration = global_end_time - global_start_time
    print(f"Global duration: {global_duration:.4f} seconds")
    # Convert the duration to minutes and seconds.
    m, s = divmod(global_duration, 60)
    h, m = divmod(m, 60)
    print(f"Global duration: {h:.0f}h {m:.0f}m {s:.0f}s")


def main():
    config = merge_config_with_cli_args(default_config)

    use_local_cluster = config.get('use_local_cluster', True)

    n_cores = int(os.getenv("NUMBER_OF_CPUS") or cpu_count())

    client = None
    cluster = None
    distributor = None

    n_workers = max(config.get('workers', 1), n_cores//2)
    memory = config.get('memory', '4GB')

    print(f"n_workers: {n_workers}, memory: {memory}, n_cores: {n_cores}")

    # increase dask resilience
    # dask.config.set({"distributed.comm.timeouts.connect": "60s"})
    # dask.config.set({'distributed.scheduler.allowed-failures': 10})

    # set local directory for dask workers
    # dask.config.set({'temporary-directory': '/tmp'})

    if use_local_cluster:

        # cluster = LocalCluster(
        #     n_workers=n_workers,
        #     memory_limit=memory,
        # )
        # client = Client(cluster)
        client = Client(memory_limit=memory)
        cluster = client.cluster

        # cluster.adapt(minimum=n_workers, maximum=n_cores-4)
        cluster.adapt(minimum=10, maximum=n_cores-4)

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
        cluster.adapt(minimum=10, maximum=n_cores-4)

        # create a dask client
        client = Client(cluster)


    distributor = ClusterDaskDistributor(cluster)

    print("client: ", client)
    print("cluster: ", cluster)

    # scale the dask workers
    # cluster.scale(n_cores-4)
    cluster.scale(n_workers//3)

    print("Waiting for workers to be ready")
    time.sleep(5)
    print("Workers ready: ", len(client.scheduler_info().get("workers")))

    target_label = config.get('target_label')

    subset = config.get('dataset_subset')

    chunk_size = config.get('chunk_size', 1000)
    print("chunk_size: ", chunk_size)

    print(f"Extracting features from the {target_label} dataset")

    print(f"Extracting features from the {subset} subset")

    try:
        feature_engineering(
            data_root_dir='datasets',
            target_label=target_label,
            subset=subset,
            client=client,
            cluster=cluster,
            distributor=distributor,
            chunk_size=chunk_size,
        )
    except Exception as e:
        print("Exception: ", e)


    # # close the dask client
    client.close()
    cluster.close()

    print("Done!")

if __name__ == '__main__':
    main()
