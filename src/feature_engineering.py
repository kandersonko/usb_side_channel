#!/usr/bin/env python3

import numpy as np
import pandas as pd
import pickle as pkl
from pathlib import Path
import time

from utils.data import load_data, preprocess_features

import matplotlib.pyplot as plt


import datetime

import lightning.pytorch as pl

#from imblearn.under_sampling import RandomUnderSampler

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.distribution import MultiprocessingDistributor
from tsfresh.feature_extraction import feature_calculators, settings

from tsfresh.feature_selection.relevance import calculate_relevance_table

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
from config import default_config, merge_config_with_cli_args


def feature_engineering(X, y, target_label, output_file, dataset, subset, config):
    benchmarking = config.get('benchmarking', False)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    ids = np.repeat(np.arange(X.shape[0]), X.shape[1])
    times = np.tile(np.arange(X.shape[1]), X.shape[0])
    values = X.flatten()  # Flatten X to get a single long list of values

    # Create a DataFrame suitable for tsfresh
    df = pd.DataFrame({'id': ids, 'time': times, 'value': values})
    print(df.head())

    # dask.config.set({'distributed.scheduler.allowed-failures': 50})
    # dask.config.set({'distributed.scheduler.worker-ttl': None})

    cluster = None
    distributor = None
    data = None
    if not benchmarking:
        data = dd.from_pandas(df, npartitions=10)
        data = data.compute()
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
        distributor = ClusterDaskDistributor(cluster)
    else:
        data = df

    plot_data = []



    print("Extracting features")

    disable_progressbar = False if benchmarking else True

    start_time = time.time()

    features = extract_features(
        data,
        column_id="id",
        column_sort="time",
        column_value="value",
        distributor=distributor,
        default_fc_parameters=settings.EfficientFCParameters(),
        disable_progressbar=disable_progressbar,
    )

    print("features shape:", features.shape)


    imputed_features = impute(features)
    print("imputed_features shape:", imputed_features.shape)

    # select relevant features
    # features.fillna(0, inplace=True)

    # selected_features = select_features(imputed_features, y)
    # print("selected_features shape:", selected_features.shape)

    # y_series = pd.Series(y, index=selected_features.index)

    # relevance_table = calculate_relevance_table(selected_features, y_series)
    # relevance_table = relevance_table[relevance_table.relevant]

    # # Sort the relevance table
    # relevance_table_sorted = relevance_table.sort_values(by='p_value')

    # # Select the top k features
    # # k = 180  # Number of features to keep
    # # top_features = relevance_table_sorted.head(k)['feature'].values
    # top_features = relevance_table_sorted['feature'].values

    # # Filter the extracted features to keep only the top k features
    # features_filtered = selected_features[top_features]
    features_filtered = imputed_features
    y_series = pd.Series(y, index=imputed_features.index)

    print("features_filtered shape:", features_filtered.shape)

    end_time = time.time()
    duration = end_time - start_time

    plot_data.append({"name": "tsfresh", "task": "feature engineering", "dataset": f"{dataset}", "duration": duration})

    if not benchmarking:
        # save the plot data
        plot_data = pd.DataFrame(plot_data)
        plot_data.to_csv(f'measurements/{dataset}-{subset}-tsfresh-feature-engineering-duration.csv')

        print(features_filtered.head())

        # save the features with the labels concatenated
        features_filtered[target_label] = y_series.loc[features_filtered.index]
        features_filtered.to_csv(output_file, index=False)



def main():
    config = merge_config_with_cli_args(default_config)


    pl.seed_everything(config['seed'], workers=True)

    target_label = config['target_label']
    dataset_name = config['dataset']
    data_dir = config['data_dir']
    dataset_subset = config['subset']
    if dataset_subset not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid subset: {dataset_subset}. Must be one of 'train', 'val', or 'test'.")

    dataset = np.load(f"{data_dir}/{dataset_name}-{target_label}.npz", allow_pickle=True)
    X_train = dataset[f'X_train']
    y_train = dataset[f'y_train']
    X_val = dataset[f'X_val']
    y_val = dataset[f'y_val']
    X_test = dataset[f'X_test']
    y_test = dataset[f'y_test']

    target_names = dataset['target_names']

    feature_engineering(X=X_train, y=y_train, target_label=target_label, output_file=f"{data_dir}/{dataset_name}-{target_label}-{dataset_subset}_tsfresh.csv", dataset=dataset_name, subset=dataset_subset, config=config)

if __name__ == '__main__':
    main()
