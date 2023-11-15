import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from os import cpu_count
from sklearn.linear_model import Perceptron
from .StockDataset import StockDataset
from .download_data import download_data
from .load_data import get_data_tensor, get_train_test_datasets


class TrainModels:
    def __init__(self, seq_len: int=100):
        """
        Args:
            seq_len (int): How many days of data to return before the target.
        """
        self.seq_len = seq_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fn = nn.MSELoss()

        # Get train and test datasets for training models
        master_tensor = get_data_tensor(relative_change=True)
        self.train_dataset, self.test_dataset = get_train_test_datasets(
                data_tensor=master_tensor,
                seq_len=seq_len,
                train_split=0.8)

        # Print out statistics
        print(f"Master Tensor Shape: {master_tensor.shape}")
        print(f"Number of Training Examples: {len(self.train_dataset)}")
        print(f"Number of Testing Examples: {len(self.test_dataset)}")
        print(f"Training device: {self.device}")

    def train_perceptron(self):
        """
        Returns:
            A fitted sklearn.linear_model.Perceptron instance.
        """
        clf = Perceptron()

        # Retrieve LARGE tensors of every single training and testing instances
        X, y = self.train_dataset.get_inputs_and_targets()
        X_test, y_test = self.test_dataset.get_inputs_and_targets()

        # Flatten date and feature dimensions of inputs because sklearn only
        # supports 2D tensors as inputs
        X = X.flatten(start_dim=1, end_dim=-1)
        X_test = X_test.flatten(start_dim=1, end_dim=-1)

        # Make target tensors 1D for sklearn as well
        y.squeeze_()
        y_test.squeeze()

        # Convert labels to 1 if positive, else 0 (limitation of sklearn only
        # accepting int labels)
        y = (y > 0).int()
        y_test = (y_test > 0).int()

        # Train Perceptron
        clf.fit(X, y)
        print(clf.n_iter_)
        print("Average days increasing in value: ", clf.predict(X_test).mean())

        return clf

    def train_RNN(self,
                  hidden_size: int=64,
                  num_layers: int=2,
                  nonlinearity: str="tanh",
                  bias: bool=True,
                  dropout: float=0,
                  batch_size: int=1,
                  lr: float=0.001):
        """
        Args:
            hidden_size (int): Number of features in each hidden state.
            num_layers (int): Number of recurrent layers. Any number greater
                than 1 results in a "stacked RNN".
            nonlinearity (str): Either "tanh" or "relu".
            bias (bool): If bias weights should be used in every layer.
            dropout (float): If non-zero, introduces Dropout layer on the
                outputs of each RNN layer expect the last layer, with dropout
                probability equal to this value.
            batch_size (int): How many samples per batch to load.
            lr (float): Learning rate.

        Returns:
            Trained instance of torch.nn.RNN model.
        """
        # Get train and test dataloaders
        train_dataloader = DataLoader(dataset=self.train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=cpu_count())
        test_dataloader = DataLoader(dataset=self.test_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=cpu_count())

        # Create RNN instance
        rnn_model = nn.RNN(input_size=self.train_dataset[0][0].shape[-1],
                     hidden_size=hidden_size,
                     num_layers=num_layers,
                     nonlinearity=nonlinearity,
                     bias=bias,
                     batch_first=True,
                     dropout=dropout)

        # Create optimizer
        opt = torch.optim.Adam(params=rnn_model.parameters(), lr=lr)

    def train_LSTM(self):
        # TODO: Train and return a LSTM model
        clf = nn.LSTM(input_size=246, hidden_size=1, num_layers=13010)

    def train_transformer(self):
        # TODO: Train and return a Transformer model
        clf = nn.Transformer()
