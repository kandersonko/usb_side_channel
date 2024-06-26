import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split

import torch
import lightning.pytorch as pl

from config import default_config, merge_config_with_cli_args

from models.autoencoders import Autoencoder
from models.utils import evaluate_detection
from dataset import extract_segments, SegmentedSignalDataModule, encode_dataset_in_batches, extract_features


def main():
    config = merge_config_with_cli_args(default_config)
    if config['task'] is None:
        raise ValueError("Provide a task")

    task = config['task']

    if config['model_path'] is None:
        raise ValueError("Provide a model path")


    pl.seed_everything(config['seed'], workers=True)

    print("loading the model")
    # load plot data file (`plot_data.csv`)
    best_model_path = config['model_path']

    model = Autoencoder(**config)
    print(model)

    model.load_state_dict(torch.load(best_model_path)['state_dict'])

    if task == "identification":
        config['target_label'] = "category"
        config['num_classes'] = 5
        title = 'Identification'
    elif task == "detection":
        config['target_label'] = "class"
        config['num_classes'] = 2
        title = 'Anomaly Detection'
    else:
        raise ValueError("Provide a valid task")

    print("config: ", config)
    # load the dataset
    print("Setting up the dataset")
    data_module = SegmentedSignalDataModule(**config)
    data_module.setup()

    target_names = data_module.target_names

    print("Extracting the segments")
    X_train, y_train, X_test, y_test = extract_segments(data_module)

    # extract the features
    print("Extracting features")
    X_train_encoded, y_train, X_test_encoded, y_test = extract_features(model, data_module)


    # print("Loading the dataset")
    # # load the train and test data with the encoded features
    # X_train_encoded = np.load("data/X_train_encoded.npy")
    # y_train = np.load("data/y_train.npy")
    # X_test_encoded = np.load("data/X_test_encoded.npy")
    # y_test = np.load("data/y_test.npy")
    # target_names = np.load("data/target_names.npy", allow_pickle=True)


    print("Training the classifier")

    classifier = RandomForestClassifier(max_depth=10, random_state=42, n_jobs=-1)


    # accuracy, report = evaluate_detection(classifier, X_train_encoded, y_train, X_test_encoded, y_test, target_names)
    classifier.fit(X_train_encoded, y_train)
    yhat = classifier.predict(X_test_encoded)
    accuracy = accuracy_score(y_test, yhat)
    report = classification_report(y_test, yhat, target_names=target_names)
    print(report)
    # save the classification report
    with open(f"results/{task}-classification_report.txt", "w") as f:
        f.write(report)

    #plot the confusion matrix
    display = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test_encoded,
        y_test,
        cmap="RdYlGn",
    )
    plt.xticks(rotation=45, ticks=range(0, len(target_names)), labels=target_names)
    plt.yticks(rotation=45, ticks=range(0, len(target_names)), labels=target_names)
    # save figure
    plt.savefig(f"results/{task}-confusion_matrix.png", dpi=300, bbox_inches="tight")

    # Predict probabilities for each class
    y_proba = classifier.predict_proba(X_test_encoded)

    # Number of classes
    n_classes = y_proba.shape[1]
    print("n_classes: ", n_classes)

    # Initialize dictionaries for ROC AUC metrics
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Plotting setup
    colors = ['blue', 'red', 'green', 'orange', 'purple'] # Adjust if more classes
    plt.figure()

    # Binary classification
    if n_classes != 5:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label='ROC curve (area = %0.2f) for anomaly/normal' % roc_auc)

    # Multi-class classification
    else:
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test == i, y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            # retrieve the target name
            target_name = target_names[i]
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'.format(target_name, roc_auc[i]))

    # Plot for both cases
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title} ROC AUC')
    plt.legend(loc="lower right")
    # save figure
    plt.savefig(f"results/{task}-roc_auc.png", dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    main()
