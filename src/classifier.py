import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset, random_split

from torchmetrics import Accuracy

from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning.pytorch as pl

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.utilities.model_summary import ModelSummary

from config import default_config as config

from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split


from models.classifiers import LSTMClassifier
from config import default_config, merge_config_with_cli_args

from sklearn.ensemble import RandomForestClassifier


def make_plots(config, classifier, X_train, y_train, X_test, y_test, target_names, dataset='raw'):
    task = config['task']
    model = config['classifier']

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


    print("Training the classifier")

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)

    yhat = None
    y_proba = None
    if model == "random_forest":
        classifier.fit(X_train, y_train)

        yhat = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)

    elif model == "lstm":

        # move the data to the GPU

        config['sequence_length'] = X_train.shape[1]
        classifier = LSTMClassifier(**config)
        print(classifier)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        print("after converting to tensors")
        print("X_train shape: ", X_train.shape)
        print("y_train shape: ", y_train.shape)
        print("X_test shape: ", X_test.shape)
        print("y_test shape: ", y_test.shape)


        # split the training dataset into train and validation
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=config['val_split'], random_state=config['seed'])

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        # print the shapes
        print("train_dataset shape: ", len(train_dataset))
        print("val_dataset shape: ", len(val_dataset))
        print("test_dataset shape: ", len(test_dataset))


        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

        early_stopping = EarlyStopping(
            config['monitor_metric'], patience=config['early_stopping_patience'], verbose=False, mode='min', min_delta=0.0)
        learning_rate_monitor = LearningRateMonitor(
            logging_interval='epoch', log_momentum=True)
        learning_rate_finder = LearningRateFinder()

        callbacks = [
            early_stopping,
            learning_rate_monitor,
            # checkpoint_callback,
            learning_rate_finder,
        ]
        torch.set_float32_matmul_precision('medium')
        trainer = pl.Trainer(
            # accumulate_grad_batches=config['ACCUMULATE_GRAD_BATCHES'],
            num_sanity_val_steps=0,
            max_epochs=config['max_epochs'],
            min_epochs=config['min_epochs'],
            accelerator="gpu",
            devices=-1,
            strategy='ddp',
            # logger=wandb_logger,
            callbacks=callbacks,
            precision="32-true",
            # precision="16-mixed",
            # precision=32,
            # default_root_dir=config['CHECKPOINT_PATH'],
        )

        trainer.fit(classifier, train_dataloader, val_dataloader)

        # save the model
        torch.save(classifier.state_dict(), f"results/{model}-{task}-{dataset}-model.pth")

        # get the predictions list
        # y_proba = trainer.predict(classifier, test_dataloader)
        y_proba = classifier(X_test)
        y_proba = y_proba.detach().cpu().numpy()
        y_proba = y_proba.reshape(-1, config['num_classes'])
        yhat = np.argmax(y_proba, axis=1)
        print("y_proba shape: ", y_proba.shape)
        print("yhat shape: ", yhat.shape)


    accuracy = accuracy_score(y_test, yhat)
    report = classification_report(y_test, yhat, target_names=target_names)
    print(report)
    # save the classification report
    with open(f"results/{model}-{task}-{dataset}-classification_report.txt", "w") as f:
        f.write(report)

        # #plot the confusion matrix
        # display = ConfusionMatrixDisplay.from_estimator(
        #     classifier,
        #     X_test,
        #     y_test,
        #     cmap="RdYlGn",
        # )
        #plot the confusion matrix
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        yhat,
        cmap="RdYlGn",
    )
    plt.xticks(rotation=45, ticks=range(0, len(target_names)), labels=target_names)
    plt.yticks(rotation=45, ticks=range(0, len(target_names)), labels=target_names)
    # save figure
    plt.savefig(f"results/{model}-{task}-{dataset}-confusion_matrix.png", dpi=300, bbox_inches="tight")

    # Predict probabilities for each class

    # Number of classes
    n_classes = y_proba.shape[1]
    # print("n_classes: ", n_classes)

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
    plt.savefig(f"results/{model}-{task}-{dataset}-roc_auc.png", dpi=300, bbox_inches="tight")

def main():

    config = merge_config_with_cli_args(default_config)

    pl.seed_everything(config['seed'], workers=True)

    # if config['features'] is None:
    #     raise ValueError("Provide a model path")

    classifier_name = config.get('classifier')
    if classifier_name is None:
        raise ValueError("Provide the classifier name (missing argument --classifier) e.g. random_forest or lstm")

    method = config.get('method')
    if method is None:
        raise ValueError("Provide the method name (missing argument --method) e.g. mean or median")

    print("Setting up the model")
    classifier = None
    if classifier_name == 'random_forest':
        classifier = RandomForestClassifier(max_depth=10, random_state=42, n_jobs=-1)
        # load the features
    elif classifier_name == 'lstm':
        classifier = LSTMClassifier(**config)

    # load the features
    print("Loading the features")

    if method == 'tsfresh':
        features = pd.read_csv('data/features_tsfresh.csv')
        print("features shape: ", features.shape)
        print("features columns: ", features.columns)
        print("features head: ", features.head())

        # get the features and labels
        y_category = features['category'].values
        y_category = y_category.astype(np.int64)
        y_class = features['class'].values
        y_class = y_class.astype(np.int64)
        X = features.drop(['category', 'class'], axis=1).values
    elif method == 'encoder':
        X_category_train_encoded = np.load("data/X_category_train_encoded.npy")
        X_class_train_encoded = np.load("data/X_class_train_encoded.npy")
        y_category_train = np.load("data/y_category_train.npy")
        y_class_train = np.load("data/y_class_train.npy")
        X_category_test_encoded = np.load("data/X_category_test_encoded.npy")
        X_class_test_encoded = np.load("data/X_class_test_encoded.npy")
        y_category_test = np.load("data/y_category_test.npy")
        y_class_test = np.load("data/y_class_test.npy")
        category_target_names = np.load("data/category_target_names.npy", allow_pickle=True)
        class_target_names = np.load("data/class_target_names.npy", allow_pickle=True)

        # identification
        # make_plots(
        #     config,
        #     classifier,
        #     X_train,
        #     y_category_train,
        #     X_test,
        #     y_category_test,
        #     category_target_names,
        #     dataset='raw',
        # )
        y_train = None
        y_test = None
        target_names = None
        task = config['task']
        if task == 'identification':
            X_train = X_category_train_encoded
            X_test = X_category_test_encoded
            y_train = y_category_train
            y_test = y_category_test
            target_names = category_target_names
        elif task == 'detection':
            X_train = X_class_train_encoded
            X_test = X_class_test_encoded
            y_train = y_class_train
            y_test = y_class_test
            target_names = class_target_names

        make_plots(
            config,
            classifier,
            X_train,
            y_train,
            X_test,
            y_test,
            target_names,
            dataset='features',
        )



if __name__ == '__main__':
    main()
