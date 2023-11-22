#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import Accuracy

from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning.pytorch as pl

from einops.layers.torch import Rearrange

from config import default_config as config


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
            **kwargs,
    ):
        super().__init__()
        self.monitor = monitor_metric
        self.learning_rate_patience = learning_rate_patience
        hidden_size = bottleneck_dim
        input_size = sequence_length
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        num_layers = num_lstm_layers
        self.num_layers = num_lstm_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        # self.dropout = nn.Dropout(dropout)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)

        self.example_input_array = torch.rand(self.batch_size, sequence_length)


        self.save_hyperparameters()

    def forward(self, x):
        x = x.unsqueeze(1)
        # Set initial hidden and cell states
        # make sure to initialize them on the same device as the model
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        # out = self.dropout(out)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        x = x.squeeze(1)
        return out


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(y_hat, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_hat, y))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        self.log('test_acc', self.accuracy(y_hat, y))
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
