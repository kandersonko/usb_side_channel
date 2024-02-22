#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import re

parser = argparse.ArgumentParser(description='Create tables from the results of the experiments')
parser.add_argument('-i', '--input', type=Path, help='Path to the results directory', required=True)
# add measurements directory
parser.add_argument('-m', '--measurements', type=Path, help='Path to the measurements directory', default='measurements')
parser.add_argument('-o', '--output', type=Path, help='Path to the output directory', default='tables')
args = parser.parse_args()

best_data = []
measurement_data = []
table_data = []

def create_classification_table(data_dir, dataset_name):
    files = list(Path(data_dir).glob(f'*-{dataset_name}-*'))
    files = sorted(files, key=lambda x: x.stem.split('-')[0]+x.stem.split('-')[4]+x.stem.split('-')[5])
    current_model = None
    data = []
    for file in files:
        # extract the model name, method, dataset, dataset_name from the filename where {model}-{task}-{method}-{dataset}-{dataset_name}-confusion_matrix.png"
        # format the model name
        model = file.stem.split('-')[0]
        task = file.stem.split('-')[1]
        method = file.stem.split('-')[2]
        dataset = file.stem.split('-')[3]
        # format the dataset name
        dataset_name = file.stem.split('-')[4]
        # image path
        image_path = file
        data_row = {
            'model': model,
            'task': task,
            'method': method,
            'dataset': dataset,
            'dataset_name': dataset_name,
        }
        # get number of folds using regex from the

        if 'classification_report' in file.name:
            # display the classification report
            classification_report = file.read_text()
            # get number of folds using regex from the classification report content
            # on the line `Number of folds: 10`
            num_folds = re.findall(r'Number of folds: (\d+)', classification_report)[0]
            data_row['num_folds'] = int(num_folds)
            scores = list(filter(lambda x: len(x.strip()), classification_report.split('\n')))[-1]
            score_values = re.split(r'\s+', scores)
            precision = float(score_values[2])
            recall = float(score_values[3])
            f1_score = float(score_values[4])
            data_row['Precision'] = precision
            data_row['Recall'] = recall
            data_row['F1 Score'] = f1_score
            data.append(data_row)

    df = pd.DataFrame(data)
    print(df.head())
    # rename the model column values
    df['Model'] = df['model'].str.replace('_', ' ').str.title()
    df.rename(columns={'num_folds': 'K-fold', 'method': 'Extractor'}, inplace=True)
    # rename 'encoder' to 'autoencoder'
    df['Extractor'] = df['Extractor'].str.replace('encoder', 'autoencoder')
    df['Extractor'] = df['Extractor'].str.replace('_', ' ').str.title()

    extractors = df['Extractor'].unique()

    for extractor in extractors:
        frame = df[df['Extractor'] == extractor]
        frame = frame.groupby(['Model']).agg({'Precision': 'mean', 'Recall': 'mean', 'F1 Score': 'mean'}).reset_index()[['Model', 'Precision', 'Recall', 'F1 Score']]
        print(frame.head())
        # save the latex table to a file
        output_dir = args.output
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'{dataset_name}-{extractor.lower()}-classification_report.tex'
        # remove previous file
        if output_file.exists():
            output_file.unlink()
        frame.to_latex(output_file, index=False, escape=False, float_format="%.2f", caption=f"Average precision, recall, f1 score for each model on {dataset_name.replace('_', ' ').title()} using {extractor.lower()} features", label=f"tab:{dataset_name}-{extractor.lower()}-classification_report", bold_rows=True, column_format="lccc", position='H')

        frame['Dataset'] = dataset_name
        table_data.append(frame)

        # find the best model
        best_model = frame.iloc[frame['F1 Score'].idxmax()]
        best_model['Extractor'] = extractor
        best_model['Dataset'] = dataset_name
        best_data.append(best_model)


    # # get the latex table from the dataframe
    # # save the latex table to a file
    # output_dir = args.output
    # output_dir.mkdir(parents=True, exist_ok=True)
    # output_file = output_dir / f'{dataset_name}-classification_report.tex'

    # # remove previous file
    # if output_file.exists():
    #     output_file.unlink()

    # df[['Model', 'Extractor', 'Precision', 'Recall', 'F1 Score']].to_latex(output_file, index=False, escape=False, float_format="%.2f", caption=f"Average Precision, Recall, F1 Score for each model on {dataset_name.replace('_', ' ').title()}", label=f"tab:{dataset_name}-classification_report", bold_rows=True, column_format="lcccc", position='h')


def create_measurement_tables(data_dir, dataset_name):
    files = list(Path(data_dir).glob(f'*-{dataset_name}-*.csv'))
    dfs = [pd.read_csv(file) for file in files]
    df = pd.concat(dfs)
    frame = df.groupby(['name', 'task', 'method'])[['name', 'task', 'method', 'duration']].agg({'duration': 'mean'})

    frame.reset_index(inplace=True)
    print(frame.columns)
    frame.columns = ['Model', 'Task', 'Method', 'Duration']
    # change 'encoder' to 'autoencoder' in method column
    frame['Duration'] = frame['Duration'].apply(lambda x: f"{x:.5f}")
    frame['Model'] = frame['Model'].str.replace('_', ' ').str.title()
    frame['Task'] = frame['Task'].str.replace('_', ' ').str.title()
    frame['Method'] = frame['Method'].str.replace('_', ' ').str.title()
    frame['Method'] = frame['Method'].str.replace('encoder', 'AE')
    frame['Task'] = frame['Task'].str.replace('Inference', 'Infer.')
    frame['Task'] = frame['Task'].str.replace('Training', 'Train')

    tasks = frame['Task'].unique()

    # Pivot the DataFrame to have separate columns for Inference Time and Training Time
    df_pivoted = frame.pivot(index='Model', columns=['Task', 'Method'], values='Duration').reset_index()
    df_pivoted.columns = df_pivoted.columns.map(lambda x: f"{x[0].title()} {x[1]}" if x[1] else x[0])
    # df_pivoted.columns.name = None  # Remove the name of the columns index
    # df_pivoted.rename(columns={'Inference': 'Inference (s)', 'Training': 'Training (s)'}, inplace=True)


    print(df_pivoted.head())
    # save the latex table to a file
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'{dataset_name}-measurement_report.tex'
    # remove previous file
    if output_file.exists():
        output_file.unlink()
    df_pivoted.to_latex(output_file, index=False, escape=False, float_format="%.5f", caption=f"Average inference and training time for each model on {dataset_name.replace('_', ' ').title()}", label=f"tab:{dataset_name}-measurement_report", bold_rows=True, column_format="lcccccc", position='h')

    df_pivoted['Dataset'] = dataset_name
    measurement_data.append(df_pivoted)


def create_feature_extraction_table(data_dir):
    files = list(Path(data_dir).glob(f'*.csv'))
    files = [f for f in files if 'feature-extraction' in f.name or 'feature-engineering' in f.name]

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        print(df.head())
        df['phase'] = f.stem.split('-')[1]
        dfs.append(df)

    df = pd.concat(dfs)

    frame = df.groupby(['name', 'dataset'])['duration'].sum().reset_index()

    print(frame.head())
    print(frame.columns)

    frame.columns = ['Model', 'Dataset', 'Duration']
    # change 'encoder' to 'autoencoder' in method column
    frame['Duration'] = frame['Duration'].apply(lambda x: f"{x:.5f}")
    frame['Model'] = frame['Model'].str.replace('_', ' ').str.title()
    frame['Dataset'] = frame['Dataset'].str.replace('_', ' ').str.title()


    # Pivot the DataFrame to have separate columns for Inference Time and Training Time
    df_pivoted = frame.pivot(index='Model', columns=['Dataset'], values='Duration').reset_index()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'feature_extraction_report.tex'

    df_pivoted.to_csv(output_dir / 'feature_extraction_report.csv', index=False)

    df_pivoted.to_latex(output_file, index=False, escape=False, float_format="%.5f", caption=f"Average feature extraction time for each model on each dataset", label=f"tab:feature_extraction_report", bold_rows=True, column_format="lcccccc", position='h')
    print(df_pivoted.head())
    return df_pivoted

def main():
    data_dir = args.input
    datasets = ['dataset_a', 'dataset_b', 'dataset_c1', 'dataset_c2', 'dataset_d1', 'dataset_d2']

    # for dataset_name in datasets:
    #     print(f"Creating table for {dataset_name}")
    #     create_classification_table(data_dir, dataset_name)
    #     print()

    #     print(f"Creating measurement table for {dataset_name}")
    #     create_measurement_tables(args.measurements, dataset_name)
    #     print()
    #     print()

    print(f"Creating feature extraction tables")
    create_feature_extraction_table(args.measurements)
    print()
    print()



    # table_df = pd.concat(table_data)

    # measurement_df = pd.concat(measurement_data)
    # print("Measurement data:")
    # print(measurement_df.head())
    # # merge the dataframes on the model and dataset name
    # merged_df = pd.merge(table_df, measurement_df, on=['Model', 'Dataset'])
    # print("Merged data:")
    # print(merged_df.head())
    # # save the merged dataframe to csv
    # output_dir = args.output
    # output_dir.mkdir(parents=True, exist_ok=True)
    # output_file = output_dir / f'tables.csv'
    # print(f"Saving the tables to models to {output_file}")
    # merged_df.to_csv(output_file, index=False)
    # print(f"Saved the tables to {output_file}")
    # print()

    # best_df = pd.DataFrame(best_data)
    # print("Best models:")
    # print(best_df.head())

if __name__ == '__main__':
    main()
