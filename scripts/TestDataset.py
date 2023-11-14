import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Perceptron
from utils.StockDataset import StockDataset
from download_data import download_data
from load_data import get_data_tensor, get_train_test_datasets


class TestLearningModels:
    def __init__(self, seq_len: int=100):
        self.seq_len = seq_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
        clf = Perceptron()
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

        clf.fit(X, y)
        print(clf.n_iter_)
        print("Average days increasing in value: ", clf.predict(X_test).mean())

    def train_RNN(self):
        clf = nn.RNN(input_size=246, hidden_size=1, num_layers=13010)
        X, y = self.get_params(self.train_dataset)
        out, hn = clf(X, y)

    def train_LSTM(self):
        clf = nn.LSTM(input_size=246, hidden_size=1, num_layers=13010)
        X, y = self.get_params(self.train_dataset)
        out, hn = clf(X, y)

    def train_transformer(self):
        clf = nn.Transformer()
        X, y = self.get_params(self.train_dataset)
        out, hn = clf(X, y)
        pass


TestLearningModels(seq_len=100)
