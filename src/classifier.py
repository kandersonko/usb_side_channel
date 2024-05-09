import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from timeit import timeit
from datetime import datetime

from pathlib import Path

from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

from lightning.pytorch.loggers import WandbLogger
import wandb
from wandb import AlertLevel

from tsfresh import select_features

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset, random_split, Subset

from torchmetrics import Accuracy

from torch.nn import DataParallel

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import WeightedRandomSampler

import lightning.pytorch as pl

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Timer
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.utilities.model_summary import ModelSummary

from lightning.pytorch.loggers import TensorBoardLogger

from config import default_config as config

from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE

from models.classifiers import LSTMClassifier, PyTorchClassifierWrapper
from config import default_config, merge_config_with_cli_args

from dataset import compute_class_weights

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

torch.multiprocessing.set_sharing_strategy('file_system')

from utilities import RankedLogger, task_wrapper, instantiate_callbacks, instantiate_loggers

log = RankedLogger(__name__, rank_zero_only=True)

FONT_SIZE = 24
FONT_WEIGHT = 'normal'
FONT_FAMILY = 'serif'


def update_matplotlib_config(font_size=FONT_SIZE, font_weight=FONT_WEIGHT, font_family=FONT_FAMILY, dpi=300):
    # FONT_SERIF = 'Times'

    # Adjust global font size and weight
    # plt.rcParams.update({'font.size': FONT_SIZE, 'font.weight': FONT_WEIGHT, 'axes.labelweight': FONT_WEIGHT, 'axes.titleweight': FONT_WEIGHT, 'figure.dpi': 600, 'font.family': FONT_FAMILY})
    plt.rcParams.update({'font.size': font_size, 'font.weight': font_weight, 'axes.labelweight': font_weight, 'axes.titleweight': font_weight, 'figure.dpi': dpi, 'font.family': font_family})

def plot_roc_auc(y_test, y_proba, target_names, model, dataset_name, dataset, task, method):
    update_matplotlib_config(font_size=12)
    # Number of classes
    # n_classes = y_proba.shape[1]
    n_classes = len(target_names)
    log.info(f"n_classes: {n_classes}")

    fpr = np.array([])
    tpr = np.array([])
    thresholds = np.array([])

    # Plotting setup
    colors = ['blue', 'red', 'green', 'orange', 'purple'] # Adjust if more classes
    plt.figure()

    # Binary classification
    if n_classes == 1 or n_classes == 2:
        # fpr, tpr, t = roc_curve(y_test, y_proba[:, 1])
        fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label='ROC AUC curve (area = %0.2f) for anomaly/normal' % roc_auc)

    # Multi-class classification
    else:
        # Initialize dictionaries for ROC AUC metrics
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            # fpr[i], tpr[i], _ = roc_curve(y_test == i, y_proba[:, i])
            fpr[i], tpr[i], _ = roc_curve(y_test == i, y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            # retrieve the target name
            target_name = target_names[i].replace("_", " ").capitalize()
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                    label='ROC AUC curve of class {0} (area = {1:0.2f})'.format(target_name, roc_auc[i]))


    # save the roc_auc as npz
    np.savez(f"results/{model}-{task}-{method}-{dataset}-{dataset_name}-roc_auc.npz", fpr=fpr, tpr=tpr, thresholds=thresholds, roc_auc=roc_auc)
    # with open(f"results/{model}-{task}-{method}-{dataset}-{dataset_name}-roc_auc.npz", "wb") as f:
        # np.savez(f, fpr, tpr, roc_auc)

    # title = f"{target_label.capitalize()} {task} ROC AUC [Model: {model}, Dataset: {dataset_name}]"
    # Plot for both cases
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title(title)
    plt.legend(loc="lower right")
    # save figure
    plt.savefig(f"results/{model}-{task}-{method}-{dataset}-{dataset_name}-roc_auc.png", dpi=300, bbox_inches="tight")


def plot_confusion_matrix(y_test, y_pred, y_proba, target_names, target_label, model, dataset_name, dataset, task, method):
    update_matplotlib_config(font_size=28)
    accuracy = accuracy_score(y_test, y_pred)
    if len(target_names) == 2 or len(target_names) == 1:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=target_names)
    log.info(f"Model: {model}")
    log.info(f"Number of folds: {config['kfold']}")
    log.info(f"Accuracy: {accuracy:.2f}")
    log.info(f"ROC AUC: {roc_auc:.2f}")
    log.info(f"\n{report}")
    with open(f"results/{model}-{task}-{method}-{dataset}-{dataset_name}-classification_report.txt", "w") as f:
        # write the dataset name and model name
        f.write(f"Model: {model}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Number of folds: {config['kfold']}\n")
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"ROC AUC: {roc_auc:.2f}\n")
        f.write("\n")
        f.write(report)

    print(f"Saving the classification report to results/{model}-{task}-{method}-{dataset}-{dataset_name}-classification_report.txt")

    # save the report as npz
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    with open(f"results/{model}-{task}-{method}-{dataset}-{dataset_name}-classification_report.npz", "wb") as f:
        np.savez(f, report)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="RdYlGn",
    )
    # increase the size of the font and make them
    # plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})
    plt.xticks(rotation=45, ticks=range(0, len(target_names)), labels=target_names)
    plt.yticks(rotation=45, ticks=range(0, len(target_names)), labels=target_names)
    # save figure
    # title = f"{target_label.capitalize()} {task} Confusion Matrix [Model: {model}, Dataset: {dataset_name}]"
    # plt.title(title)
    plt.savefig(f"results/{model}-{task}-{method}-{dataset}-{dataset_name}-confusion_matrix.png", dpi=300, bbox_inches="tight")



def train_lstm(classifier, X_train, y_train, X_val, y_val, X_test, y_test, config, task, class_weights=None):
    class_weights = None

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)


    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)


    class_counts = torch.bincount(y_train)
    class_weights = 1. / class_counts
    samples_weights = class_weights[y_train]

    # Create the sampler
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_dataloader = DataLoader(train_dataset, batch_size=config.datasets.batch_size, shuffle=False, num_workers=config.datasets.num_workers, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=config.datasets.batch_size, shuffle=False, num_workers=config.datasets.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=config.datasets.batch_size, shuffle=False, num_workers=config.datasets.num_workers)


    logger = instantiate_loggers(config.logger)
    callbacks = instantiate_callbacks(config.callbacks)

    # model_checkpoint_idx = -1
    # for idx, callback in enumerate(callbacks):
    #     if isinstance(callback, pl.callbacks.ModelCheckpoint):
    #         model_checkpoint_idx = idx
    #         break

    # date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    # if model_checkpoint_idx != -1:
    #     callbacks[model_checkpoint_idx].filename = config.model_name+'-{epoch:02d}-{val_loss:.2f}-'+date_time

    trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    torch.set_float32_matmul_precision('medium')
    if torch.cuda.is_available():
        log.info(f"Available GPU devices: {torch.cuda.device_count()}")


    # measure train time
    # start_time = time.time()
    trainer.fit(classifier, train_dataloader, val_dataloader)



    log.info(classifier)
    # end_time = time.time()
    # duration = end_time - start_time
    timer = None
    for callback in trainer.callbacks:
        if isinstance(callback, pl.callbacks.Timer):
            timer = callback
    duration = timer.time_elapsed("train")

    log.info(f"\nConfig:\n {OmegaConf.to_yaml(config)}")
    log.info(f"Method: {config.method}")
    log.info(f"Model: {config.model_name}")
    log.info(f"Dataset: {config.datasets.dataset}")
    log.info(f"Target label: {config.datasets.target_label}")
    log.info(f"Task: {task}")
    log.info(f"X_train shape: {X_train.shape}")
    log.info(f"y_train shape: {y_train.shape}")
    log.info(f"X_test shape: {X_test.shape}")
    log.info(f"y_test shape: {y_test.shape}")

    best_model_path = None
    if trainer.is_global_zero:
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_model_path = Path(best_model_path).relative_to(Path.cwd())
        log.info(f"Best model path: {best_model_path}")
    # log.info the shapes
    log.info(f"train_dataset length: {len(train_dataset)}")
    log.info(f"val_dataset length: {len(val_dataset)}")
    log.info(f"test_dataset length: {len(test_dataset)}")

    # return trainer.checkpoint_callback.best_model_path, duration
    return duration, trainer, best_model_path


def get_predictions(classifier, X, y, config, model="ml"):
    # Define 10-fold cross-validation
    kfold = StratifiedKFold(n_splits=config['kfold'], shuffle=True, random_state=config['seed'])

    y_pred = None
    y_prob = None  # Initialize y_prob

    # Perform cross-validation and get predictions
    # if model == "dl":
    y_pred = cross_val_predict(classifier, X, y, cv=kfold, method='predict', verbose=3)
    if hasattr(classifier, "predict_proba"):
        y_prob = cross_val_predict(classifier, X, y, cv=kfold, method='predict_proba') # Get probabilities for the positive class
    elif hasattr(classifier, "decision_function"):
        y_prob = cross_val_predict(classifier, X, y, cv=kfold, method='decision_function')
    else:
        raise ValueError("Classifier does not support probability predictions.")

    return y_pred, y_prob


def measure_inference_time(classifier, X_test, model, dataset_name, task, method):
    sample = X_test[0].reshape(1, -1)
    plot_data = []
    yhat = None
    start_time = time.time()
    yhat = classifier.predict(sample)
    end_time = time.time()
    duration = end_time - start_time
    log.info(f"Inference time: {duration:.4f} seconds")
    plot_data.append({"name": model, "task": "inference", "dataset": dataset_name, "method": method, "duration": duration})
    # save the plot data
    plot_data = pd.DataFrame(plot_data)
    plot_data.to_csv(f"measurements/{model}-{task}-{method}-{dataset_name}-inference-duration.csv")
    return duration


def make_plots(config, target_label, X_train, y_train, X_val, y_val, X_test, y_test, target_names, method, dataset='features', dataset_name='dataset_a'):
    pl.seed_everything(config['seed'])
    task = config['task']
    model = config['classifier']
    num_classes = len(target_names)

    if task not in ["identification", "detection"]:
        raise ValueError("Provide a valid task")

    log.info("Training the classifier")
    log.info(f"dataset: {dataset_name}")
    log.info(f"target_label: {target_label}")
    log.info(f"X_train shape: {X_train.shape}")
    log.info(f"y_train shape: {y_train.shape}")
    log.info(f"X_test shape: {X_test.shape}")
    log.info(f"y_test shape: {y_test.shape}")

    # replace _ with space and capitalize the first letter
    target_names = [target_name.replace("_", " ").capitalize() for target_name in target_names]

    yhat = None
    y_proba = None
    classifier = None
    if model in ["random_forest", "decision_tree", "KNN", "gradient_boosting", "SVC"]:
        ml_classifiers = {
            "random_forest": RandomForestClassifier(random_state=config['seed'], max_depth=10, n_jobs=-1),
            # "random_forest": RandomForestClassifier(random_state=config['seed'], max_depth=10),
            "decision_tree": DecisionTreeClassifier(random_state=config['seed']),
            "KNN": KNeighborsClassifier(n_neighbors=4),
            "gradient_boosting": GradientBoostingClassifier(),
            "SVC": SVC(probability=True, random_state=config['seed']),
        }
        classifier = ml_classifiers[model]
        # measure train time
        plot_data = []
        start_time = time.time()
        log.info(f'Before training: {start_time}')
        classifier.fit(X_train, y_train)
        end_time = time.time()
        log.info(f'After training: {end_time}')
        duration = end_time - start_time
        plot_data.append({"name": model, "task": "training", "dataset": dataset_name, "method": method, "duration": duration})
        # save the plot data
        plot_data = pd.DataFrame(plot_data)
        plot_data.to_csv(f"measurements/{model}-{task}-{method}-{dataset}-{dataset_name}-training-duration.csv")


        # yhat = classifier.predict(X_test)
        yhat, y_proba = get_predictions(classifier, X_test, y_test, config=config)

        measure_inference_time(classifier, X_test, model, dataset_name, task, method)
        log.info(f"Training time: {duration:.4f} seconds")
        log.info(f"Inference time: {duration:.4f} seconds")
        # y_proba = classifier.predict_proba(X_test)

    else:

        plot_data = []
        classifier = None

        # config['lstm_input_dim'] = X_train.shape[1]
        config.model.num_classes = len(np.unique(y_train))
        # config['lstm_output_dim'] = len(np.unique(y_train))
        config.model.sequence_length = X_train.shape[1]

        classifier = LSTMClassifier(**config.model)

        # train the lstm classifier
        # best_model_path, duration = train_lstm(classifier, X_train, y_train, X_val, y_val, X_test, y_test, config, task)
        duration, trainer, best_model_path = train_lstm(classifier, X_train, y_train, X_val, y_val, X_test, y_test, config, task)

        if trainer.is_global_zero:
            # best_model_path = trainer.checkpoint_callback.best_model_path

            classifier = LSTMClassifier.load_from_checkpoint(best_model_path)
            # classifier = LSTMClassifier.load_from_checkpoint(best_model_path)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            classifier = classifier.to(device)
            classifier = PyTorchClassifierWrapper(classifier, nn.CrossEntropyLoss(), torch.optim.Adam(classifier.parameters()), epochs=config.trainer.max_epochs)

            yhat, y_proba = get_predictions(classifier, X_test, y_test, config=config, model="dl")

            report = classification_report(y_test, yhat, target_names=target_names, output_dict=True)
            f1_score = report['weighted avg']['f1-score']
            log.info(f"F1 score: {f1_score:.4f}")
            log.info(f"Precision: {report['weighted avg']['precision']:.4f}")
            log.info(f"Recall: {report['weighted avg']['recall']:.4f}")

            plot_data.append({"name": "lstm", "task": "training", "dataset": dataset_name, "method": method, "duration": duration})
            # save the plot data
            plot_data = pd.DataFrame(plot_data)
            log.info(f"Training time: {duration:.4f} seconds")
            plot_data.to_csv(f"measurements/{model}-{task}-{method}-{dataset}-{dataset_name}-lstm-training-duration.csv")

            measure_inference_time(classifier, X_test, model, dataset_name, task, method)



    plot_roc_auc(y_test, y_proba, target_names, model, dataset_name, dataset, task, method)
    plot_confusion_matrix(y_test, yhat, y_proba, target_names, target_label, model, dataset_name, dataset, task, method)



def tune(config, X_train, y_train, X_val, y_val, X_test, y_test, task, target_names):

    # add noise to the data
    # std_dev = 0.005  # Standard deviation of the noise
    # noise = np.random.normal(0, std_dev, size=X_train.shape)

    model_name = config.get('model_name')
    model = LSTMClassifier(**config.model)

    plot_data = []

    # best_model_path, duration = train_lstm(model, X_train, y_train, X_val, y_val, X_test, y_test, config, task)
    duration, trainer, best_model_path = train_lstm(model, X_train, y_train, X_val, y_val, X_test, y_test, config, task)

    if trainer.is_global_zero:
        log.info(f"Best model path: {best_model_path}")
        # best_model_path = trainer.checkpoint_callback.best_model_path
        plot_data.append({"name": model_name, "task": "training", "dataset": config.datasets.dataset, "method": config.method, "duration": duration})

        # save the plot data
        plot_data = pd.DataFrame(plot_data)
        plot_data.to_csv(f"measurements/{model_name}-{task}-{config.method}-{config.datasets.dataset}-training-duration.csv")

        # if model_name == 'pure-lstm':
        #     base_model = PureLSTMClassifier.load_from_checkpoint(best_model_path)
        # elif model_name == 'lstm-encoder':

        # best_model_path = trainer.checkpoint_callback.best_model_path
        base_model = LSTMClassifier.load_from_checkpoint(best_model_path)

        classifier = PyTorchClassifierWrapper(base_model, nn.CrossEntropyLoss(), torch.optim.Adam(base_model.parameters()), epochs=config.trainer.max_epochs)

        y_pred, y_proba = get_predictions(classifier, X_test, y_test, config=config, model="dl")

        accuracy = accuracy_score(y_test, y_pred)
        # Generate classification report
        report = classification_report(y_test, y_pred, target_names=target_names)
        log.info(f"Number of folds: {config['kfold']}")
        log.info(f"\n{report}")


        dataset_name = config.datasets.dataset
        method = config.method
        model = config.model_name
        log.info(f"Training duration: {duration:.4f} seconds")
        inference_duration = measure_inference_time(classifier, X_test, model, dataset_name, task, method)

        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        f1_score = report['weighted avg']['f1-score']
        log.info(f"F1 score: {f1_score:.4f}")
        log.info(f"Precision: {report['weighted avg']['precision']:.4f}")
        log.info(f"Recall: {report['weighted avg']['recall']:.4f}")
    # log to wandb

        logger = trainer.logger.experiment
        if hasattr(logger, "log"):
            logger.log({"f1_score": report['weighted avg']['f1-score']})
            logger.log({"precision": report['weighted avg']['precision']})
            logger.log({"recall": report['weighted avg']['recall']})
            logger.log({"accuracy": accuracy})
            logger.log({"training_duration": duration})
            logger.log({"inference_duration": inference_duration})

        if config.logger.wandb and f1_score > 0.99:
            log.info(f"F1 score is greater than 0.99 (value: {f1_score}).")
            wandb.alert(
                title="F1 score is greater than 0.95",
                text=f"F1 score value is {f1_score}. \nModel: {model_name}, Dataset: {config.datasets.dataset}, Method: {config['method']}, duration: {duration:.4f} seconds",
                level=AlertLevel.WARN,
                wait_duration=300,
            )


def process_anomaly_detection_dataset(dataset_name, X_train, y_train, X_val, y_val, X_test, y_test, target_names):

    if dataset_name == 'dataset_d1':
        # transform the label to 'anomaly' and 'normal' where the OMG device is the anomaly
        anomaly_class = target_names.tolist().index('teensyduino')
        y_train[y_train != anomaly_class] = 0
        y_train[y_train == anomaly_class] = 1
        y_val[y_val != anomaly_class] = 0
        y_val[y_val == anomaly_class] = 1
        y_test[y_test != anomaly_class] = 0
        y_test[y_test == anomaly_class] = 1

        target_names = np.array(['normal', 'anomaly'], dtype=object)

    elif dataset_name == 'dataset_d2':
        # transform the label to 'anomaly' and 'normal' where the OMG device is the anomaly
        anomaly_class = target_names.tolist().index('OMG')
        y_train[y_train != anomaly_class] = 0
        y_train[y_train == anomaly_class] = 1
        y_val[y_val != anomaly_class] = 0
        y_val[y_val == anomaly_class] = 1
        y_test[y_test != anomaly_class] = 0
        y_test[y_test == anomaly_class] = 1

        target_names = np.array(['normal', 'anomaly'], dtype=object)
    else:
        raise ValueError("Provide a valid dataset name")

    return X_train, y_train, X_val, y_val, X_test, y_test, target_names


@task_wrapper
def classify(config):

    pl.seed_everything(config['seed'])

    tuning = config.get('tuning', False)

    method = config.get('method')

    target_label = 'category' if config.datasets.dataset == 'dataset_a' else 'device'
    config.datasets.target_label = target_label
    log.info(f"Target label: {target_label}")

    # load the features
    log.info("Loading the features")

    data_dir = config.datasets.data_dir

    # task = config['task']
    task=None
    if config.datasets.dataset in ['dataset_d1', 'dataset_d2']:
        task = 'detection'
    else:
        task = 'identification'
    config.task = task

    dataset_name = config.datasets.dataset
    target_label = config.datasets.target_label

    if method == 'raw':

        # dataset = np.load(f"{data_dir}/{subset}_dataset.npz", allow_pickle=True)
        dataset = np.load(f"{data_dir}/{dataset_name}-{target_label}.npz", allow_pickle=True)
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_val = dataset['X_val']
        y_val = dataset['y_val']
        X_test = dataset['X_test']
        y_test = dataset['y_test']
        target_names = dataset['target_names']

        config.datasets.sequence_length = X_train.shape[1]
        config.model.sequence_length = X_train.shape[1]

        if dataset_name in ['dataset_d1', 'dataset_d2']:
            X_train, y_train, X_val, y_val, X_test, y_test, target_names = process_anomaly_detection_dataset(dataset_name, X_train, y_train, X_val, y_val, X_test, y_test, target_names)

        if tuning:
            tune(config, X_train, y_train, X_val, y_val, X_test, y_test, task, target_names)
        else:
            make_plots(
                config,
                target_label,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                target_names,
                dataset='raw',
                dataset_name=dataset_name,
                method=method,
            )

    if method == 'tsfresh':
        # smote = SMOTE()
        subset = None
        if task == 'identification':
            subset = 'category'
        elif task == 'detection':
            subset = 'class'

        raw_dataset = np.load(f"{data_dir}/{dataset_name}-{target_label}.npz", allow_pickle=True)

        train = pd.read_csv(f"{data_dir}/{dataset_name}-{target_label}-train_tsfresh.csv")
        y_train = train[target_label].values
        X_train = train.drop(target_label, axis=1).values
        log.info(f"X_train shape: {X_train.shape}")

        val = pd.read_csv(f"{data_dir}/{dataset_name}-{target_label}-val_tsfresh.csv")
        y_val = val[target_label].values
        X_val = val.drop(target_label, axis=1).values
        log.info(f"X_val shape: {X_val.shape}")

        test = pd.read_csv(f"{data_dir}/{dataset_name}-{target_label}-test_tsfresh.csv")
        y_test = test[target_label].values
        X_test = test.drop(target_label, axis=1).values
        log.info(f"X_test shape: {X_test.shape}")

        target_names = raw_dataset['target_names']

        config.datasets.sequence_length = X_train.shape[1]
        config.model.sequence_length = X_train.shape[1]

        if dataset_name in ['dataset_d1', 'dataset_d2']:
            X_train, y_train, X_val, y_val, X_test, y_test, target_names = process_anomaly_detection_dataset(dataset_name, X_train, y_train, X_val, y_val, X_test, y_test, target_names)

        if tuning:
            tune(config, X_train, y_train, X_val, y_val, X_test, y_test, task, target_names)
        else:
            make_plots(
                config,
                target_label,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                target_names,
                dataset='tsfresh',
                dataset_name=dataset_name,
                method=method,
            )

    elif method == 'encoder':

        # dataset = np.load(f"{data_dir}/{subset}_dataset.npz", allow_pickle=True)
        dataset = np.load(f"{data_dir}/{dataset_name}-{target_label}_features.npz", allow_pickle=True)
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_val = dataset['X_val']
        y_val = dataset['y_val']
        X_test = dataset['X_test']
        y_test = dataset['y_test']
        target_names = dataset['target_names']

        config.datasets.sequence_length = X_train.shape[1]
        config.model.sequence_length = X_train.shape[1]

        if dataset_name in ['dataset_d1', 'dataset_d2']:
            X_train, y_train, X_val, y_val, X_test, y_test, target_names = process_anomaly_detection_dataset(dataset_name, X_train, y_train, X_val, y_val, X_test, y_test, target_names)


        if tuning:
            tune(config, X_train, y_train, X_val, y_val, X_test, y_test, task, target_names)
        else:
            make_plots(
                config,
                target_label,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                target_names,
                dataset='features',
                dataset_name=dataset_name,
                method=method,
            )


    return dict(metrics={}), dict(config=config)

@hydra.main(version_base="1.3", config_path="./configs", config_name="config.yaml")
def main(config: DictConfig):
    # log.info(OmegaConf.to_yaml(config))
    classify(config)

if __name__ == '__main__':
    main()
