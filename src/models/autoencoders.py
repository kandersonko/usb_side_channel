#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import Accuracy

from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning.pytorch as pl

# from einops.layers.torch import Rearrange

from config import default_config as config

class PureAutoencoder(pl.LightningModule):
    def __init__(
            self,
            bottleneck_dim,
            batch_size,
            learning_rate,
            sequence_length,
            learning_rate_patience,
            conv1_out_channels,
            conv2_out_channels,
            num_lstm_layers,
            dropout,
            monitor_metric,
            **kwargs,
    ):
        super().__init__()

        # save all hyperparameters
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.learning_rate_patience = learning_rate_patience
        self.dropout = dropout
        self.monitor = monitor_metric
        self.bottleneck_dim = bottleneck_dim
        self.batch_size = batch_size


        # conv1d expects (batch, channels, seq_len)
        self.example_input_array = torch.rand(self.batch_size, sequence_length)


        self.encoder = CNNLSTMEncoder(
            sequence_length=sequence_length,
            num_features=1,
            hidden_size=bottleneck_dim,
            conv1_out_channels=conv1_out_channels,
            conv2_out_channels=conv2_out_channels,
            num_layers=num_lstm_layers,
            dropout=dropout,
        )

        self.decoder = CNNLSTMDecoder(
            sequence_length=sequence_length,
            hidden_size=bottleneck_dim,
            num_layers=num_lstm_layers,
            conv1_out_channels=conv2_out_channels, # we reverse the channels
            conv2_out_channels=conv1_out_channels,
            dropout=dropout,
        )

        # Loss Functions
        self.reconstruction_loss_fn = nn.MSELoss()


    def forward(self, x):
        # import pdb; pdb.set_trace()
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)

        # Compute losses
        total_loss = self.reconstruction_loss_fn(x_hat, x)

        self.log('train_loss', total_loss, sync_dist=True, prog_bar=True)
        return total_loss


    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        total_loss = self.reconstruction_loss_fn(x_hat, x)

        self.log('val_loss', total_loss, sync_dist=True, prog_bar=True)

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


class Autoencoder(pl.LightningModule):
    def __init__(
            self,
            num_classes,
            reconstruction_loss_weight,
            classification_loss_weight,
            bottleneck_dim,
            batch_size,
            learning_rate,
            sequence_length,
            learning_rate_patience,
            conv1_out_channels,
            conv2_out_channels,
            num_lstm_layers,
            dropout,
            monitor_metric,
            **kwargs,
    ):
        super().__init__()

        # save all hyperparameters
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.learning_rate_patience = learning_rate_patience
        self.dropout = dropout
        self.monitor = monitor_metric
        self.bottleneck_dim = bottleneck_dim
        self.batch_size = batch_size

        self.num_classes = num_classes
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.classification_loss_weight = classification_loss_weight


        # conv1d expects (batch, channels, seq_len)
        self.example_input_array = torch.rand(self.batch_size, sequence_length)

        # Initialize accuracy metrics for multiclass classification
        self.train_accuracy = Accuracy(num_classes=self.num_classes, task="multiclass")
        self.val_accuracy = Accuracy(num_classes=self.num_classes, task="multiclass")


        # hidden_dims = [512, 256 ]

        # Encoder
        activation_fn = nn.LeakyReLU()


        # self.encoder = LSTMEncoder(input_size=1, hidden_size=bottleneck_dim, num_layers=2)
        # self.decoder = LSTMDecoder(input_size=1, hidden_size=bottleneck_dim, num_layers=2, sequence_length=sequence_length)


        self.encoder = CNNLSTMEncoder(
            sequence_length=sequence_length,
            num_features=1,
            hidden_size=bottleneck_dim,
            conv1_out_channels=conv1_out_channels,
            conv2_out_channels=conv2_out_channels,
            num_layers=num_lstm_layers,
            dropout=dropout,
        )

        self.decoder = CNNLSTMDecoder(
            sequence_length=sequence_length,
            hidden_size=bottleneck_dim,
            num_layers=num_lstm_layers,
            conv1_out_channels=conv2_out_channels, # we reverse the channels
            conv2_out_channels=conv1_out_channels,
            dropout=dropout,
        )

        # Classifier Head
        self.classifier = nn.Linear(bottleneck_dim, num_classes)

        # Loss Functions
        self.reconstruction_loss_fn = nn.MSELoss()
        # if self.class_weights is None:
        #     self.classification_loss_fn = nn.CrossEntropyLoss()
        # else:
        #     self.classification_loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        self.classification_loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # import pdb; pdb.set_trace()
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Classify based on the encoded representation
        classification = self.classifier(encoded)
        return decoded, classification

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_hat, y_hat = self.forward(x)

        # Compute losses
        reconstruction_loss = self.reconstruction_loss_fn(x_hat, x) * self.reconstruction_loss_weight
        classification_loss = self.classification_loss_fn(y_hat, y) * self.classification_loss_weight
        total_loss = reconstruction_loss + classification_loss

        # Update accuracy metric
        self.train_accuracy(y_hat, y)

        self.log('train_loss', total_loss, sync_dist=True, prog_bar=True)
        self.log('train_accuracy', self.train_accuracy, sync_dist=True, prog_bar=True)
        self.log('train_recon_loss', reconstruction_loss, sync_dist=True)
        self.log('train_class_loss', classification_loss, sync_dist=True)
        return total_loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat, y_hat = self.forward(x)
        reconstruction_loss = self.reconstruction_loss_fn(x_hat, x) * self.reconstruction_loss_weight
        classification_loss = self.classification_loss_fn(y_hat, y) * self.classification_loss_weight

        total_loss = reconstruction_loss + classification_loss

        # Update accuracy metric
        self.val_accuracy(y_hat, y)

        self.log('val_loss', total_loss, sync_dist=True, prog_bar=True)

        self.log('val_accuracy', self.val_accuracy, sync_dist=True, prog_bar=True)

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


class LinearEncoder(nn.Module):
    def __init__(self, bottleneck_dim, activation_fn=nn.ReLU(True), input_dim=1000, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim

        layers = [input_dim] + hidden_dims + [bottleneck_dim]

       # Create the encoder layers
        encoder_layers = []
        for i in range(len(layers) - 1):
            encoder_layers.append(nn.Linear(layers[i], layers[i+1]))
            encoder_layers.append(activation_fn)

        self.net = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.net(x)



class LinearDecoder(nn.Module):
    def __init__(self, bottleneck_dim, activation_fn=nn.ReLU(True), input_dim=1000, hidden_dims=[512, 256, 128]):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim

        layers = [input_dim] + hidden_dims + [bottleneck_dim]

        decoder_layers = []

        # reverse the layer dims

        for i in range(len(layers) - 1, 0, -1):
            decoder_layers.append(nn.Linear(layers[i], layers[i-1]))
            # Use ReLU for all layers except for the last one
            if i > 1:
                decoder_layers.append(activation_fn)
                # else:
                # Typically the last layer would have a sigmoid if we expect the output to be [0,1]
                # decoder_layers.append(nn.Sigmoid())
                # decoder_layers.append(nn.Tanh())

        self.net = nn.Sequential(*decoder_layers)


    def forward(self, x):
        return self.net(x)



class ConvEncoder(nn.Module):
    def __init__(self, channels, bottleneck_dim, activation_fn=nn.ReLU(True)):
        super().__init__()
        self.channels = channels
        self.bottleneck_dim = bottleneck_dim
        self.dropout = config['DROPOUT']

        self.batch_size = config['BATCH_SIZE']

        # Convolutional layers
        conv_layers = []
        for i in range(len(channels) - 1):
            conv_layers.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1))
            conv_layers.append(nn.BatchNorm1d(channels[i + 1]))
            conv_layers.append(activation_fn)
            if i % 2 == 0:
                conv_layers.append(nn.Dropout(self.dropout))
                self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate the flattened feature dimension
        self.feature_dim = self._get_conv_output_dim()

        # Bottleneck linear layer
        self.bottleneck = nn.Sequential(
            nn.Linear(self.feature_dim, bottleneck_dim),
            activation_fn,
            nn.Dropout(self.dropout),
        )

    def _get_conv_output_dim(self):
        # Empirically determine the output dimension of conv layers for the linear layer
        with torch.no_grad():
            dummy_input = torch.zeros(self.batch_size, self.channels[0], 1000, device=self.conv_layers[0].weight.device)
            dummy_output = self.conv_layers(dummy_input)
            return int(torch.numel(dummy_output) / dummy_output.shape[0])

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv_layers(x)
        # x = nn.Flatten(1)(x)  # Flatten
        x = x.view(x.size(0), -1)  # Flatten
        # Linear bottleneck


class ConvDecoder(nn.Module):
    def __init__(self, channels, bottleneck_dim, activation_fn=nn.ReLU(True)):
        super().__init__()
        self.channels = channels
        self.bottleneck_dim = bottleneck_dim

        self.batch_size = config['BATCH_SIZE']
        self.dropout = config['DROPOUT']

        # Convolutional layers
        conv_layers = []
        for i in range(len(channels) - 1, 0, -1):
            conv_layers.append(nn.ConvTranspose1d(channels[i], channels[i - 1], kernel_size=3, stride=2, padding=1, output_padding=1))
            conv_layers.append(nn.BatchNorm1d(channels[i - 1]))
            conv_layers.append(activation_fn)
            if i % 2 == 0:
                conv_layers.append(nn.Dropout(self.dropout))
                conv_layers.pop()  # remove the last ReLU for the output layer

        self.conv_layers = nn.Sequential(*conv_layers)

        # The first layer of the decoder is a linear layer
        self.bottleneck = nn.Sequential(
            nn.Linear(bottleneck_dim, channels[-1] * self._get_conv_output_dim()),
            activation_fn,
            nn.Dropout(self.dropout),
        )


    def _get_conv_output_dim(self):
        # Empirically determine the output dimension of conv layers for the linear layer
        with torch.no_grad():
            dummy_input = torch.zeros(self.batch_size, self.channels[-1], 1000, device=self.conv_layers[0].weight.device)
            dummy_output = self.conv_layers(dummy_input)
            return int(torch.numel(dummy_output) / dummy_output.shape[0] / self.channels[-1])

    def forward(self, x):
        # set breakpoint
        # import pdb

        # pdb.set_trace()
        x = self.bottleneck(x)  # Linear bottleneck
        # x = nn.Unflatten(1, (self.channels[-1], self._get_conv_output_dim()))(x)  # Unflatten
        x = x.view(x.size(0), self.channels[-1], -1)  # Unflatten
        # x = x.view(x.size(0), self.channels[-1], self._get_conv_output_dim())
        x = self.conv_layers(x)  # Convolutional layers
        # Remove channel dimension



class LSTMEncoder(nn.Module):
    def __init__(self, input_size=1, sequence_length=config['sequence_length'], hidden_size=config['bottleneck_dim'], num_layers=1):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        x = x.reshape(-1, self.sequence_length, 1)
        _, (hidden, _) = self.lstm(x)
        features = hidden[-1, :, :]



class LSTMDecoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=config['bottleneck_dim'],
                 num_layers=1, sequence_length=config['sequence_length']):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, input_size)


    def forward(self, hidden):
        # `hidden` is the feature vector from the encoder and has a shape of: (batch_size, hidden_size)
        # We need to add a dimension to match the LSTM's expected input shape, which is:
        # (batch_size, sequence_length, input_size)
        # We initialize the sequence to zeros
        batch_size = hidden.size(0)
        decoder_input = torch.zeros(batch_size, self.sequence_length, self.hidden_size, device=hidden.device)

        # We now need to initialize the hidden and cell state for the LSTM
        # Since we are only taking the last layer's hidden state from the encoder, we need to
        # repeat it `num_layers` times for the initial hidden state
        h_0 = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=hidden.device)

        # Now we can pass the input and the initial states to the LSTM
        lstm_out, _ = self.lstm(decoder_input, (h_0, c_0))

        # Pass the output of the LSTM to the linear layer to get our decoded output
        lstm_out = lstm_out.contiguous().view(batch_size * self.sequence_length, self.hidden_size)
        decoded = self.linear(lstm_out)

        # Finally, we reshape it to get our final output shape (batch_size, sequence_length)
        decoded = decoded.view(batch_size, self.sequence_length)




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


class CNNLSTMEncoder(nn.Module):
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

        # Linear layer to get to the desired hidden_size
        # self.fc = nn.Linear(cnn_output_size * hidden_size, hidden_size) # for the non-attention layer
        #
        # self.fc = nn.Linear(hidden_size, hidden_size) # for the attention layer
        self.fc = nn.Linear(hidden_size*2, hidden_size) # multiply by 2 because of bidirectional

        # add dropout
        self.dropout = nn.Dropout(dropout)

        # Attention layer
        self.attention = Attention(hidden_size*2) # multiply by 2 because of bidirectional


    def forward(self, x):
        # import pdb; pdb.set_trace()
        # Apply CNN layers
        x = x.unsqueeze(1)  # Add a channel dimension

        # x = torch.relu(self.bn1(self.conv1(x)))
        x = self.conv1(x)
        # x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool(x)

        # x = torch.relu(self.bn2(self.conv2(x)))
        x = self.conv2(x)
        # x = self.bn2(x)
        x = torch.relu(x)

        x = self.pool(x)

        x = self.dropout(x)

        # Reshape for LSTM
        x = x.transpose(1, 2)  # Swap channel and sequence_length dimensions
        x, _ = self.lstm(x)

        # Apply attention
        x, attention_weights = self.attention(x)

        # Reshape and apply linear layer
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x



class CNNLSTMDecoder(nn.Module):
    def __init__(self,
                 sequence_length,
                 hidden_size,
                 num_layers,
                 conv1_out_channels, # we reverse the channels
                 conv2_out_channels,
                 dropout,
                 **kwargs,
                 ):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Define LSTM layers
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        # Define upsampling and CNN layers
        self.upsample1 = nn.Upsample(size=sequence_length // 2)
        # self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=conv1_out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv1d(in_channels=hidden_size*2, out_channels=conv1_out_channels, kernel_size=3, stride=1, padding=1) # multiply by 2 because of bidirectional

        self.bn1 = nn.BatchNorm1d(conv1_out_channels)

        self.upsample2 = nn.Upsample(size=sequence_length)
        self.conv2 = nn.Conv1d(in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=3, stride=1, padding=1)

        self.bn2 = nn.BatchNorm1d(conv2_out_channels)

        self.conv3 = nn.Conv1d(in_channels=conv2_out_channels, out_channels=1, kernel_size=3, stride=1, padding=1)

        # Linear layer to reshape the input to the LSTM
        # self.fc = nn.Linear(hidden_size, hidden_size * (sequence_length // 4))
        self.fc = nn.Linear(hidden_size, hidden_size * (sequence_length // 4))

        # add dropout
        self.dropout = nn.Dropout(dropout)

        # Attention layer
        # self.attention = Attention(hidden_size)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # Apply linear layer and reshape for LSTM
        x = self.fc(x)
        x = x.view(x.size(0), self.sequence_length // 4, self.hidden_size)

        # Apply LSTM layers
        x, _ = self.lstm(x)

        # Apply attention using encoder attention weights
        # x, _ = self.attention(x * encoder_attention_weights)

        # x = x.unsqueeze(1)  # Add channel dimension

        x = x.transpose(1, 2)  # Swap sequence_length and channel dimensions

        
        # Apply upsampling and CNN layers
        x = self.upsample1(x)
        # x = torch.relu(self.bn1(self.conv1(x)))
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.upsample2(x)
        # x = torch.relu(self.bn2(self.conv2(x)))
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)


        x = self.dropout(x)
        
        # x = torch.sigmoid(self.conv3(x))
        x = self.conv3(x)

        # Remove the channel dimension
        x = x.squeeze(1)
        return x
