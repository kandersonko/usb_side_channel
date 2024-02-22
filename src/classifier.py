import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime

from tqdm import tqdm

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

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
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

from models.classifiers import LSTMClassifier, PyTorchClassifierWrapper, PureLSTMClassifier
from config import default_config, merge_config_with_cli_args

from dataset import compute_class_weights

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

torch.multiprocessing.set_sharing_strategy('file_system')

def plot_roc_auc(y_test, y_proba, target_names, model, dataset_name, dataset, task, method):
    # Number of classes
    # n_classes = y_proba.shape[1]
    n_classes = len(target_names)
    print("n_classes: ", n_classes)

    # Initialize dictionaries for ROC AUC metrics
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Plotting setup
    colors = ['blue', 'red', 'green', 'orange', 'purple'] # Adjust if more classes
    plt.figure()

    # Binary classification
    if n_classes == 1:
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
            target_name = target_names[i].replace("_", " ").capitalize()
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'.format(target_name, roc_auc[i]))


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


def train_lstm(classifier, X_train, y_train, X_val, y_val, X_test, y_test, config, task, class_weights=None):
    config['sequence_length'] = X_train.shape[1]
    config['num_classes'] = len(np.unique(y_train))
    class_weights = None

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    print("after converting to tensors")
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)


    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # print the shapes
    print("train_dataset shape: ", len(train_dataset))
    print("val_dataset shape: ", len(val_dataset))
    print("test_dataset shape: ", len(test_dataset))

    class_counts = torch.bincount(y_train)
    class_weights = 1. / class_counts
    samples_weights = class_weights[y_train]

    # Create the sampler
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # X_test = X_test.cuda()
    # y_test = y_test.cuda()

    early_stopping = EarlyStopping(
        config['monitor_metric'], patience=config['early_stopping_patience'], verbose=False, mode='min', min_delta=0.0)
    learning_rate_monitor = LearningRateMonitor(
        logging_interval='epoch', log_momentum=True)
    learning_rate_finder = LearningRateFinder()

    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    checkpoint_callback = ModelCheckpoint(
        # or another metric such as 'val_accuracy'
        monitor=config['monitor_metric'],
        dirpath=config['checkpoint_path'],
        filename=f"{config['model_name']}-{config['dataset']}-{config['method']}" + '-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}-'+date_time,
        save_top_k=1,
        mode='min',  # 'min' for loss and 'max' for accuracy
    )

    callbacks = [
        early_stopping,
        learning_rate_monitor,
        checkpoint_callback,
        learning_rate_finder,
    ]
    torch.set_float32_matmul_precision('medium')
    if torch.cuda.is_available():
        print("Available GPU devices: ", torch.cuda.device_count())

    trainer = pl.Trainer(
        # accumulate_grad_batches=config['ACCUMULATE_GRAD_BATCHES'],
        log_every_n_steps=4,
        num_sanity_val_steps=0,
        max_epochs=config['max_epochs'],
        min_epochs=config['min_epochs'],
        accelerator="gpu",
        devices=len(list(range(torch.cuda.device_count()))),
        strategy='ddp',
        logger=config['logger'],
        callbacks=callbacks,
        precision="32-true",
        # precision="16-mixed",
        # precision=32,
        # default_root_dir=config['CHECKPOINT_PATH'],
    )

    trainer.fit(classifier, train_dataloader, val_dataloader)

    return checkpoint_callback.best_model_path


def evaluate_lstm(X_train, y_train, X_val, y_val, config, fold):
    config['sequence_length'] = X_train.shape[1]
    classifier = LSTMClassifier(**config)

    dataset_name = config['dataset']

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_val shape: ", X_val.shape)
    print("y_val shape: ", y_val.shape)


    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    class_counts = torch.bincount(y_train)
    class_weights = 1. / class_counts
    samples_weights = class_weights[y_train]

    # Create the sampler
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # X_test = X_test.cuda()
    # y_test = y_test.cuda()

    early_stopping = EarlyStopping(
        config['monitor_metric'], patience=config['early_stopping_patience'], verbose=False, mode='min', min_delta=0.0)
    learning_rate_monitor = LearningRateMonitor(
        logging_interval='epoch', log_momentum=True)
    learning_rate_finder = LearningRateFinder()

    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    checkpoint_callback = ModelCheckpoint(
        # or another metric such as 'val_accuracy'
        monitor=config['monitor_metric'],
        dirpath='./checkpoints',
        filename='classifier-' + f"{dataset_name}-{fold}" +'-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}-'+date_time,
        save_top_k=1,
        mode='min',  # 'min' for loss and 'max' for accuracy
    )
    callbacks = [
        early_stopping,
        learning_rate_monitor,
        checkpoint_callback,
        learning_rate_finder,
    ]
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        # accumulate_grad_batches=config['ACCUMULATE_GRAD_BATCHES'],
        log_every_n_steps=4,
        num_sanity_val_steps=0,
        max_epochs=config['max_epochs'],
        min_epochs=config['min_epochs'],
        accelerator="gpu",
        devices=-1,
        strategy='ddp',
        logger=config['logger'],
        callbacks=callbacks,
        precision="32-true",
        # precision="16-mixed",
        # precision=32,
        # default_root_dir=config['CHECKPOINT_PATH'],
    )

    trainer.fit(classifier, train_dataloader, val_dataloader)


    classifier = LSTMClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device)


    return classifier, val_dataloader




def get_predictions(classifier, X, y, config, model="ml"):
    # Define 10-fold cross-validation
    kfold = StratifiedKFold(n_splits=config['kfold'], shuffle=True, random_state=config['seed'])

        # create tensor from the numpy array
        # X = torch.tensor(X, dtype=torch.float32)
        # y = torch.tensor(y, dtype=torch.long)
        # move the data to the gpu
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # X = X.to(device)
        # y = y.to(device)

    y_pred = None
    y_prob = None  # Initialize y_prob

    # Perform cross-validation and get predictions
    # if model == "dl":
    y_pred = cross_val_predict(classifier, X, y, cv=kfold, method='predict', verbose=3)
    # else:
    #     y_pred = cross_val_predict(classifier, X, y, cv=kfold, method='predict', verbose=3, n_jobs=-1)

    # For models that have a decision_function or predict_proba method, use one of them to get probabilities
    if hasattr(classifier, "predict_proba"):
        y_prob = cross_val_predict(classifier, X, y, cv=kfold, method='predict_proba') # Get probabilities for the positive class
    elif hasattr(classifier, "decision_function"):
        y_prob = cross_val_predict(classifier, X, y, cv=kfold, method='decision_function')
    else:
        raise ValueError("Classifier does not support probability predictions.")

    return y_pred, y_prob



def k_fold_cross_validation(model, X, y, config, batch_size=32):
    # Assuming X and y are numpy arrays; convert them to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    print("X shape: ", X_tensor.shape)
    print("y shape: ", y_tensor.shape)

    # Define 10-fold cross-validation
    kfold = StratifiedKFold(n_splits=config['kfold'], shuffle=True, random_state=config['seed'])

    y_pred = []
    y_prob = []

    num_classes = np.unique(y).shape[0]

    fold = 0

    model = None

    for train_index, test_index in kfold.split(X, y):
        # Split the data into training and testing sets
        X_train, X_test = X_tensor[train_index], X_tensor[test_index]
        y_train, y_test = y_tensor[train_index], y_tensor[test_index]
        print(f"Fold {fold + 1} of {config['kfold']}")
        print("num_classes: ", num_classes)
        print("X_train shape: ", X_train.shape)
        print("y_train shape: ", y_train.shape)
        print("X_test shape: ", X_test.shape)
        print("y_test shape: ", y_test.shape)

        model, test_dataloader = evaluate_lstm(X_train, y_train, X_test, y_test, config, fold)

        y_pred_batch, y_prob_batch = predict(model, test_dataloader, num_classes)
        y_pred.append(y_pred_batch)
        y_prob.append(y_prob_batch)

    y_pred = np.concatenate(y_pred, axis=1)
    y_prob = np.concatenate(y_prob, axis=1)

    return y_pred, y_prob, model

def predict(model, dataloader, num_classes):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # model = DataParallel(model)

    model.eval()  # Set the model to evaluation mode

    y_proba_list = []
    with torch.no_grad():
        for batch in dataloader:
            # Move data to GPU if necessary
            X_batch, _ = batch
            X_batch = X_batch.cuda(non_blocking=True, device=device)

            # Forward pass
            y_proba_batch = model(X_batch)
            y_proba_batch = y_proba_batch.cpu().numpy()

            # Accumulate predictions
            y_proba_list.append(y_proba_batch)
        # Concatenate predictions for all batches
    y_proba = np.vstack(y_proba_list)
    y_proba = y_proba.reshape(-1, num_classes)
    yhat = np.argmax(y_proba, axis=1)
    return yhat, y_proba


def plot_confusion_matrix(y_test, y_pred, target_names, target_label, model, dataset_name, dataset, task, method):
    accuracy = accuracy_score(y_test, y_pred)
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("Number of folds: ", config['kfold'])
    print(report)
    with open(f"results/{model}-{task}-{method}-{dataset}-{dataset_name}-classification_report.txt", "w") as f:
        # write the dataset name and model name
        f.write(f"Model: {model}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Number of folds: {config['kfold']}\n")
        f.write("\n")
        f.write(report)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="RdYlGn",
    )
    plt.xticks(rotation=45, ticks=range(0, len(target_names)), labels=target_names)
    plt.yticks(rotation=45, ticks=range(0, len(target_names)), labels=target_names)
    # save figure
    # title = f"{target_label.capitalize()} {task} Confusion Matrix [Model: {model}, Dataset: {dataset_name}]"
    # plt.title(title)
    plt.savefig(f"results/{model}-{task}-{method}-{dataset}-{dataset_name}-confusion_matrix.png", dpi=300, bbox_inches="tight")


def measure_inference_time(classifier, X_test, model, dataset_name, task, method):
    sample = X_test[0].reshape(1, -1)
    plot_data = []
    # measure inference time
    start_time = time.time()
    yhat = None
    if model == "lstm":
        # sample = torch.tensor(sample, dtype=torch.float32)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # sample = sample.to(device)
        # classifier = classifier.to(device)
        # yhat = classifier(sample)
        yhat = classifier.predict(sample)
    else:
        yhat = classifier.predict(sample)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Inference time: {duration:.4f} seconds")
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

    print("Training the classifier")
    print("dataset: ", dataset_name)
    print("target_label: ", target_label)
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape)
    print("y_test shape: ", y_test.shape)

    # replace _ with space and capitalize the first letter
    target_names = [target_name.replace("_", " ").capitalize() for target_name in target_names]

    yhat = None
    y_proba = None
    classifier = None
    if model in ["random_forest", "decision_tree", "KNN", "gradient_boosting", "SVC"]:
        ml_classifiers = {
            "random_forest": RandomForestClassifier(random_state=config['seed'], max_depth=10, n_jobs=-1),
            "decision_tree": DecisionTreeClassifier(random_state=config['seed']),
            "KNN": KNeighborsClassifier(n_neighbors=4),
            "gradient_boosting": GradientBoostingClassifier(),
            "SVC": SVC(probability=True, random_state=config['seed']),
        }
        classifier = ml_classifiers[model]
        # measure train time
        start_time = time.time()
        plot_data = []

        classifier.fit(X_train, y_train)

        end_time = time.time()
        duration = end_time - start_time
        plot_data.append({"name": model, "task": "training", "dataset": dataset_name, "method": method, "duration": duration})
        # save the plot data
        plot_data = pd.DataFrame(plot_data)
        plot_data.to_csv(f"measurements/{model}-{task}-{method}-{dataset}-{dataset_name}-training-duration.csv")


        # yhat = classifier.predict(X_test)
        yhat, y_proba = get_predictions(classifier, X_test, y_test, config=config)

        measure_inference_time(classifier, X_test, model, dataset_name, task, method)
        # y_proba = classifier.predict_proba(X_test)

    elif model == "lstm":

        # measure train time
        start_time = time.time()

        plot_data = []
        classifier = None

        if config['model_path'] is not None and config['model_path'] != '':
            classifier = LSTMClassifier.load_from_checkpoint(config['model_path'])
            # move to gpu
            classifier.cuda()
            # move the features to the gpu

            # classifier.load_state_dict(torch.load(config['model_path'])['state_dict'])
        else:
            classifier = LSTMClassifier(**config)

        # train the lstm classifier
        best_model_path = train_lstm(classifier, X_train, y_train, X_val, y_val, X_test, y_test, config, task)


        end_time = time.time()
        duration = end_time - start_time

        classifier = LSTMClassifier.load_from_checkpoint(best_model_path)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        classifier = classifier.to(device)
        # predict
        # yhat, y_proba = predict(classifier, test_dataloader, config)
        # y_proba = y_proba.reshape(-1, config['num_classes'])
        # yhat = np.argmax(y_proba, axis=1)
        # yhat, y_proba, classifier = k_fold_cross_validation(classifier, X_test, y_test, config=config, batch_size=config['batch_size'])


        # move the model to the gpu
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # base_model = base_model.to(device)

        # base_model = LSTMClassifier(**config)
        classifier = PyTorchClassifierWrapper(classifier, nn.CrossEntropyLoss(), torch.optim.Adam(classifier.parameters()), epochs=config['max_epochs'])

        yhat, y_proba = get_predictions(classifier, X_test, y_test, config=config, model="dl")


        plot_data.append({"name": "lstm", "task": "training", "dataset": dataset_name, "method": method, "duration": duration})
        # save the plot data
        plot_data = pd.DataFrame(plot_data)
        plot_data.to_csv(f"measurements/{model}-{task}-{method}-{dataset}-{dataset_name}-lstm-training-duration.csv")

        measure_inference_time(classifier, X_test, model, dataset_name, task, method)


    plot_confusion_matrix(y_test, yhat, target_names, target_label, model, dataset_name, dataset, task, method)

    plot_roc_auc(y_test, y_proba, target_names, model, dataset_name, dataset, task, method)



def tune(config, X_train, y_train, X_val, y_val, X_test, y_test, task, target_names):
    print(f"Tuning the model {config['model_name']}")
    config['lstm_input_dim'] = X_train.shape[1]
    config['num_classes'] = len(np.unique(y_train))
    config['lstm_output_dim'] = len(np.unique(y_train))
    config['sequence_length'] = X_train.shape[1]
    print("num_classes: ", config['num_classes'])

    model_name = config.get('model_name')
    if model_name == 'pure-lstm':
        model = PureLSTMClassifier(**config)
    elif model_name == 'lstm-encoder':
        model = LSTMClassifier(**config)

    # model = PureLSTMClassifier(**config)

    # measure train time
    start_time = time.time()

    plot_data = []

    best_model_path = train_lstm(model, X_train, y_train, X_val, y_val, X_test, y_test, config, task)

    end_time = time.time()
    duration = end_time - start_time
    plot_data.append({"name": model_name, "task": "training", "dataset": config['dataset'], "method": config['method'], "duration": duration})

    # save the plot data
    plot_data = pd.DataFrame(plot_data)
    plot_data.to_csv(f"measurements/{model_name}-{task}-{config['method']}-{config['dataset']}-training-duration.csv")

    # base_model = LSTMClassifier.load_from_checkpoint(best_model_path)
    # base_model = PureLSTMClassifier.load_from_checkpoint(best_model_path)

    if model_name == 'pure-lstm':
        base_model = PureLSTMClassifier.load_from_checkpoint(best_model_path)
    elif model_name == 'lstm-encoder':
        base_model = LSTMClassifier.load_from_checkpoint(best_model_path)


    classifier = PyTorchClassifierWrapper(base_model, nn.CrossEntropyLoss(), torch.optim.Adam(base_model.parameters()), epochs=config['max_epochs'])

    y_pred, y_proba = get_predictions(classifier, X_test, y_test, config=config, model="dl")

    accuracy = accuracy_score(y_test, y_pred)
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("Number of folds: ", config['kfold'])
    print(report)


    dataset_name = config['dataset']
    method = config['method']
    model = config['model_name']
    print(f"Training duration: {duration:.4f} seconds")
    inference_duration = measure_inference_time(classifier, X_test, model, dataset_name, task, method)

    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    # log to wandb
    wandb.log({"f1_score": report['weighted avg']['f1-score']})
    wandb.log({"precision": report['weighted avg']['precision']})
    wandb.log({"recall": report['weighted avg']['recall']})
    wandb.log({"accuracy": accuracy})
    wandb.log({"training_duration": duration})
    wandb.log({"inference_duration": inference_duration})

    f1_score = report['weighted avg']['f1-score']

    if f1_score > 0.95:
        print(f"F1 score is greater than 0.95 (value: {f1_score}).")
        wandb.alert(
            title="F1 score is greater than 0.95",
            text=f"F1 score value is {f1_score}. \nModel: {model_name}, Dataset: {config['dataset']}, Method: {config['method']}, Using encoder: {config['use_encoder']}",
            level=AlertLevel.WARN,
            wait_duration=300,
        )


def main():

    config = merge_config_with_cli_args(default_config)

    pl.seed_everything(config['seed'])

    tuning = config.get('tuning', False)
    print("tuning: ", tuning)

    log = config.get('log', False)

    if tuning:
        config['checkpoint_path'] = 'tuning_checkpoints'
    else:
        config['checkpoint_path'] = 'checkpoints'

    logger = None
    wandb_logger = WandbLogger(project="usb_side_channel", config=config)
    if log or tuning:
        logger = wandb_logger

    config['logger'] = logger

    # if config['features'] is None:
    #     raise ValueError("Provide a model path")

    classifier_name = config.get('classifier')
    if classifier_name is None:
        raise ValueError("Provide the classifier name (missing argument --classifier) e.g. random_forest or lstm")

    method = config.get('method')
    if method is None:
        raise ValueError("Provide the method name (missing argument --method) e.g. raw or tsfresh or encoder")


    # load the features
    print("Loading the features")

    data_dir = config['data_dir']

    task = config['task']

    dataset_name = config['dataset']
    target_label = config['target_label']

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
        print("X_train shape: ", X_train.shape)

        val = pd.read_csv(f"{data_dir}/{dataset_name}-{target_label}-val_tsfresh.csv")
        y_val = val[target_label].values
        X_val = val.drop(target_label, axis=1).values
        print("X_val shape: ", X_val.shape)

        test = pd.read_csv(f"{data_dir}/{dataset_name}-{target_label}-test_tsfresh.csv")
        y_test = test[target_label].values
        X_test = test.drop(target_label, axis=1).values
        print("X_test shape: ", X_test.shape)

        target_names = raw_dataset['target_names']

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



if __name__ == '__main__':
    main()
    wandb.finish()
