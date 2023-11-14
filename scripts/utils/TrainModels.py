import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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

        # Get train and test datasets for training models
        master_tensor = get_data_tensor(relative_change=True)
        print(f"Master Tensor Shape: {master_tensor.shape}")
        self.train_dataset, self.test_dataset = get_train_test_datasets(
                data_tensor=master_tensor,
                seq_len=seq_len,
                train_split=0.8)

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

    def train_RNN(self):
        # TODO: Train and return a RNN model
        clf = nn.RNN(input_size=246, hidden_size=1, num_layers=13010)
        X, y = self.get_params(self.train_dataset)
        out, hn = clf(X, y)

    def train_LSTM(self):
        # TODO: Train and return a LSTM model
        clf = nn.LSTM(input_size=246, hidden_size=1, num_layers=13010)
        X, y = self.get_params(self.train_dataset)
        out, hn = clf(X, y)

    def train_transformer(self):
        # TODO: Train and return a Transformer model
        clf = nn.Transformer()
        X, y = self.get_params(self.train_dataset)
        out, hn = clf(X, y)
