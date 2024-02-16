#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import Accuracy

from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning.pytorch as pl

from config import default_config as config

# from models.autoencoders import CNNLSTMEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict

class PyTorchClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, criterion, optimizer, epochs=10):
        # move the model to the gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.classes_ = None  # Initialize the classes_ attribute

    def fit(self, X, y):
        # Set the classes_ attribute
        self.classes_ = torch.unique(torch.tensor(y)).numpy()

        # Convert X and y to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        # Training loop
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = self.criterion(output, y_tensor)
            loss.backward()
            self.optimizer.step()

        return self

    def predict(self, X):
        self.model.eval()
        # Turn off gradient computation
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    # Implement predict_proba if your task is classification
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = torch.softmax(self.model(X_tensor), dim=1)

        return outputs.cpu().numpy()


class PureLSTMClassifier(pl.LightningModule):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, lstm_output_dim, lstm_num_layers, lstm_dropout, lstm_bidirectional, num_classes, learning_rate, learning_rate_patience, batch_size, monitor_metric, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.monitor = monitor_metric

        self.learning_rate_patience = learning_rate_patience
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=lstm_input_dim,
                            hidden_size=lstm_hidden_dim,
                            num_layers=lstm_num_layers,
                            batch_first=True,
                            dropout=lstm_dropout, bidirectional=lstm_bidirectional)
        self.classifier = nn.Linear(lstm_hidden_dim, lstm_output_dim)
        self.loss_function = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # import pdb; pdb.set_trace()
        x = self.classifier(lstm_out)
        x = x.squeeze(1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        self.log('train_acc', self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)
        self.log('learning_rate', self.learning_rate, sync_dist=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        self.log('val_acc', self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log('test_loss', loss, sync_dist=True, prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)
        return loss


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return y_hat


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        monitor = self.monitor
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=self.learning_rate_patience, verbose=False),
            'interval': 'epoch',  # or 'step'
            'frequency': 1,
            'monitor': monitor,  # Name of the metric to monitor
        }
        return [optimizer], [scheduler]



# Create a LSTM classifier
class LSTMClassifier(pl.LightningModule):
    def __init__(
            self,
            sequence_length,
            bottleneck_dim,
            num_lstm_layers,
            num_classes,
            dropout,
            batch_size,
            learning_rate,
            monitor_metric,
            learning_rate_patience,
            conv1_out_channels,
            conv2_out_channels,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dropout = nn.Dropout(dropout)
        self.monitor = monitor_metric
        self.learning_rate_patience = learning_rate_patience
        hidden_size = bottleneck_dim
        input_size = sequence_length
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        num_layers = num_lstm_layers
        self.num_layers = num_lstm_layers
        self.batch_size = batch_size
        # self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            # num_layers=num_layers, batch_first=True)
        # use the STMEncoder
        self.encoder = LSTMEncoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_lstm_layers, dropout=dropout, sequence_length=sequence_length, num_features=1, conv1_out_channels=conv1_out_channels, conv2_out_channels=conv2_out_channels)

        self.fc = nn.Linear(hidden_size, num_classes)
        # self.dropout = nn.Dropout(dropout)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        self.example_input_array = torch.rand(self.batch_size, sequence_length)



    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.squeeze(1)
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, sync_dist=True, prog_bar=True)
        self.log('train_acc', self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        self.log('val_acc', self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss, sync_dist=True, prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)
        return loss


    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        monitor = self.monitor
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, patience=self.learning_rate_patience, verbose=False),
            'interval': 'epoch',  # or 'step'
            'frequency': 1,
            'monitor': monitor,  # Name of the metric to monitor
        }
        return [optimizer], [scheduler]


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_length, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: (batch_size, seq_length, 1)

        # Apply attention weights
        attended = attention_weights * lstm_output
        # attended shape: (batch_size, seq_length, hidden_size)

        # Sum over the sequence dimension
        output = torch.sum(attended, dim=1)
        # output shape: (batch_size, hidden_size)

        return output, attention_weights


class LSTMEncoder(nn.Module):
    def __init__(self, sequence_length,
                 num_features,
                 hidden_size,
                 conv1_out_channels,
                 num_layers,
                 conv2_out_channels,
                 dropout,
                 **kwargs,
                 ):
        super().__init__()
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Define CNN layers
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=conv1_out_channels, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm1d(conv1_out_channels)
        self.conv2 = nn.Conv1d(in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm1d(conv2_out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate the size of the features after the CNN layers
        cnn_output_size = sequence_length // 2 // 2

        # Define LSTM layers
        self.lstm = nn.LSTM(input_size=conv2_out_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        # self.lstm = nn.LSTM(input_size=conv2_out_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)

        # Linear layer to get to the desired hidden_size
        # self.fc = nn.Linear(cnn_output_size * hidden_size, hidden_size) # for the non-attention layer
        #
        # self.fc = nn.Linear(hidden_size, hidden_size) # for the attention layer
        self.fc = nn.Linear(hidden_size*2, hidden_size) # multiply by 2 because of bidirectional
        # self.fc = nn.Linear(hidden_size, hidden_size)

        # add dropout
        self.dropout = nn.Dropout(dropout)

        # Attention layer
        self.attention = Attention(hidden_size*2) # multiply by 2 because of bidirectional is True
        # self.attention = Attention(hidden_size)


    def forward(self, x):
        # import pdb; pdb.set_trace()
        # Apply CNN layers
        x = x.unsqueeze(1)  # Add a channel dimension

        x = self.conv1(x)
        # x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.dropout(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = torch.relu(x)

        x = self.pool(x)

        x = self.dropout(x)

        # Reshape for LSTM
        x = x.transpose(1, 2)  # Swap channel and sequence_length dimensions
        x, _ = self.lstm(x)

        x = self.dropout(x)

        # Apply attention
        x, attention_weights = self.attention(x)

        # Reshape and apply linear layer
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x
