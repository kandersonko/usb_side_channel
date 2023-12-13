import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from tsfresh import select_features

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset, random_split

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
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE

from models.classifiers import LSTMClassifier
from config import default_config, merge_config_with_cli_args

from dataset import compute_class_weights

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Define a function to perform predictions
def predict(model, dataloader):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model = model.to(device)
        model = DataParallel(model)

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
    return y_proba

def make_plots(config, target_label, X_train, y_train, X_val, y_val, X_test, y_test, target_names, method, dataset='features', dataset_name='dataset_a'):
    pl.seed_everything(config['seed'])
    task = config['task']
    model = config['classifier']
    num_classes = len(target_names)

    if task not in ["identification", "detection"]:
        raise ValueError("Provide a valid task")

    print("Training the classifier")

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
        classifier.fit(X_train, y_train)
        yhat = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)

    elif model == "lstm":

        # move the data to the GPU

        config['sequence_length'] = X_train.shape[1]
        if config['use_class_weights']:
            class_weights = compute_class_weights(y_train)
            classifier = LSTMClassifier(**config, class_weights=class_weights)
        else:
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

        if config['model_path'] is not None and config['model_path'] != '':
            classifier = LSTMClassifier.load_from_checkpoint(config['model_path'])
            # move to gpu
            classifier.cuda()
            # move the features to the gpu

            # classifier.load_state_dict(torch.load(config['model_path'])['state_dict'])
        else:

            early_stopping = EarlyStopping(
                config['monitor_metric'], patience=config['early_stopping_patience'], verbose=False, mode='min', min_delta=0.0)
            learning_rate_monitor = LearningRateMonitor(
                logging_interval='epoch', log_momentum=True)
            learning_rate_finder = LearningRateFinder()

            checkpoint_callback = ModelCheckpoint(
                # or another metric such as 'val_accuracy'
                monitor=config['monitor_metric'],
                dirpath='./checkpoints',
                filename='classifier-'+ task + '-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}',
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
                # logger=wandb_logger,
                logger=TensorBoardLogger("lightning_logs", name="lstm"),
                callbacks=callbacks,
                precision="32-true",
                # precision="16-mixed",
                # precision=32,
                # default_root_dir=config['CHECKPOINT_PATH'],
            )

            trainer.fit(classifier, train_dataloader, val_dataloader)


            # best_model_path = checkpoint_callback.best_model_path
            # classifier = Autoencoder.load_from_checkpoint(best_model_path)
            # classifier.load_state_dict(torch.load(best_model_path)['state_dict'])
            classifier = LSTMClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)




        # predict
        y_proba = predict(classifier, test_dataloader)
        y_proba = y_proba.reshape(-1, config['num_classes'])
        yhat = np.argmax(y_proba, axis=1)


    accuracy = accuracy_score(y_test, yhat)
    report = classification_report(y_test, yhat, target_names=target_names)
    print(report)
    # save the classification report
    with open(f"results/{model}-{task}-{method}-{dataset}-{dataset_name}-classification_report.txt", "w") as f:
        # write the dataset name and model name
        f.write(f"Model: {model}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Method: {method}\n")
        # add new line
        f.write("\n")
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
    title = f"{target_label.capitalize()} {task} Confusion Matrix [Model: {model}, Dataset: {dataset_name}]"
    plt.title(title)
    plt.savefig(f"results/{model}-{task}-{method}-{dataset}-{dataset_name}-confusion_matrix.png", dpi=300, bbox_inches="tight")

    # Predict probabilities for each class

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


    title = f"{target_label.capitalize()} {task} ROC AUC [Model: {model}, Dataset: {dataset_name}]"
    # Plot for both cases
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    # save figure
    plt.savefig(f"results/{model}-{task}-{method}-{dataset}-{dataset_name}-roc_auc.png", dpi=300, bbox_inches="tight")

def main():

    config = merge_config_with_cli_args(default_config)

    pl.seed_everything(config['seed'])

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
        dataset = np.load(f"{data_dir}/{dataset_name}.npz", allow_pickle=True)
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_val = dataset['X_val']
        y_val = dataset['y_val']
        X_test = dataset['X_test']
        y_test = dataset['y_test']
        target_names = dataset['target_names']

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

        raw_dataset = np.load(f"{data_dir}/{subset}_dataset.npz", allow_pickle=True)

        X_train = pd.read_csv(f"{data_dir}/{subset}_train_tsfresh.csv").values
        y_train = raw_dataset['y_train']
        X_val = pd.read_csv(f"{data_dir}/{subset}_val_tsfresh.csv").values
        y_val = raw_dataset['y_val']
        X_test = pd.read_csv(f"{data_dir}/{subset}_test_tsfresh.csv").values
        y_test = raw_dataset['y_test']
        target_names = raw_dataset['target_names']

        # X_train = select_features(X_train, y_train)
        # X_val = select_features(X_val, y_val)
        # X_test = select_features(X_test, y_test)


        make_plots(
            config,
            classifier,
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
        subset = None
        if task == 'identification':
            subset = 'category'
        elif task == 'detection':
            subset = 'class'

        X_train = np.load(f"{data_dir}/X_{subset}_train_encoded.npy")
        y_train = np.load(f"{data_dir}/y_{subset}_train_encoded.npy")
        X_test = np.load(f"{data_dir}/X_{subset}_test_encoded.npy")
        y_test = np.load(f"{data_dir}/y_{subset}_test_encoded.npy")

        target_names = np.load(f"{data_dir}/{subset}_target_names_encoded.npy", allow_pickle=True)

        make_plots(
            config,
            target_label,
            X_train,
            y_train,
            X_test,
            y_test,
            target_names,
            dataset='features',
            dataset_name=dataset_name,
            method=method,
        )



if __name__ == '__main__':
    main()
