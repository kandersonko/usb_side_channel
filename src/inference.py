from sklearn.ensemble import RandomForestClassifier

import numpy as np

import torch

import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner
# from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
import wandb

from models.autoencoders import Autoencoder
from models.utils import evaluate_detection
from dataset import extract_segments, SegmentedSignalDataModule, encode_dataset_in_batches, extract_features

from config import default_config, merge_config_with_cli_args


# from callbacks import InputMonitor

# path the best model path as an argument

torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    config = merge_config_with_cli_args(default_config)

    if config['model_path'] is None:
        raise ValueError("Provide a model path")

    # config['batch_size'] = 512


    pl.seed_everything(config['seed'], workers=True)

    print("Inference")
    data_module = SegmentedSignalDataModule(**config)

    print("Setting up the dataset")
    data_module.setup()

    print("Setting up the model")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # or another metric such as 'val_accuracy'
        dirpath='best_models/',
        filename='best_model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',  # 'min' for loss and 'max' for accuracy
    )

    # best_model_path = checkpoint_callback.best_model_path
    # best_model_path = 'best_models/best_model-epoch=16-val_loss=0.36.ckpt'
    # best_model_path = 'best_models/best_model-epoch=49-val_loss=0.11.ckpt'
    best_model_path = config['model_path']

    model = Autoencoder(**config)
    summary = ModelSummary(model, max_depth=-1)
    print(model)
    print(summary)

    model.load_state_dict(torch.load(best_model_path)['state_dict'])

    print("Extracting the segments")

    # Extract segments and labels from the training dataset
    X_train, y_train, X_test, y_test = extract_segments(data_module)

    target_names = data_module.target_names

    print("Evaluating the model")

    output_file_content = ""

    # training a random forest classifier without feature extraction
    classifier = RandomForestClassifier(max_depth=10, random_state=42, n_jobs=-1)
    accuracy, report = evaluate_detection(classifier, X_train, y_train, X_test, y_test, target_names)


    print("dataset shape: ", X_train.shape, y_train.shape)

    # log the results to wandb


    print()
    print("Without feature extraction")
    print("Classifier: ", classifier.__class__.__name__)
    print(f"Accuracy: {accuracy*100.0:.4f}")
    print(report)
    print()

    output_file_content += "dataset shape: " + str(X_train.shape) + " " + str(y_train.shape) + "\n"
    output_file_content += "Without feature extraction\n"
    output_file_content += "Classifier: " + str(classifier.__class__.__name__) + "\n"
    output_file_content += "Accuracy: " + str(accuracy*100.0) + "\n"
    output_file_content += str(report) + "\n"


    print("Extracting features")

    # training a random forest classifier with feature extraction
    # X_train_encoded = encode_dataset_in_batches(model, torch.tensor(X_train, dtype=torch.float32))
    # X_test_encoded = encode_dataset_in_batches(model, torch.tensor(X_test, dtype=torch.float32))

    X_train_encoded, y_train, X_test_encoded, y_test = extract_features(model, data_module)

    print("Training the classifier")

    classifier = RandomForestClassifier(max_depth=10, random_state=42, n_jobs=-1)

    accuracy, report = evaluate_detection(classifier, X_train_encoded, y_train, X_test_encoded, y_test, target_names)

    # log the results to wandb

    print("With feature extraction")
    print("Classifier: ", classifier.__class__.__name__)
    print(f"Accuracy: {accuracy*100.0:.4f}")
    print(report)

    output_file_content += "With feature extraction\n"
    output_file_content += "Classifier: " + str(classifier.__class__.__name__) + "\n"
    output_file_content += "Accuracy: " + str(accuracy*100.0) + "\n"
    output_file_content += str(report) + "\n"


    # save the output
    with open("output.txt", "w") as f:
        f.write(output_file_content)

    # save the encoded features and labels dataset to disk to the data/ folder
    # using numpy

    np.save("data/X_train_encoded.npy", X_train_encoded)
    np.save("data/y_train.npy", y_train)
    np.save("data/X_test_encoded.npy", X_test_encoded)
    np.save("data/y_test.npy", y_test)




if __name__ == '__main__':
    main()
