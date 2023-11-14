import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Perceptron
from utils.StockDataset import StockDataset
from download_data import *
from load_data import *


class TestLearningModels:
    def __init__(self, type: str, seq_len: int):
        self.master_tensor = get_data_tensor(relative_change=False)
        self.seq_len = seq_len
        print(f"Master Tensor Shape: {self.master_tensor.shape}")
        self.master_train, self.master_test = get_train_test_datasets(self.master_tensor, self.seq_len, 0.8)
        print(f"Number of Training Examples: {len(self.master_train)}")
        print(f"Number of Testing Examples: {len(self.master_test)}")
        print("GPU available?: ", torch.cuda.is_available())

        if type == "lstm":
            self.lstm()
        elif type == "rnn":
            self.rnn()
        elif type == "trans":
            self.transformer()
        elif type == "perc":
            self.perceptron()
        else:
            print("Incorrect argument used. Try either \"lstm\", \"rnn\", \"trans\", or \"perc\".")

    def get_params(self, data_tensor: torch.Tensor):
        return (data_tensor[:,1:]), (data_tensor[:,0])

    def divide(self, y):
        for i in range(1, len(y)):
            if y[i] > y[i-1]:
                y[i] = 1
            else:
                y[i] = 0
        y[0] = 0
        print(y)
        return y

    def perceptron(self):
        clf = Perceptron()
        X,y = self.get_params(self.master_train)
        X_test,y_test = self.get_params(self.master_test)
        y = self.divide(y)
        clf.fit(X,y)
        print(clf.n_iter_)
        print("Average days increasing in value: ", clf.predict(X_test).mean())

    def mlp(self):
        clf = nn.RNN(input_size=246, hidden_size=1, num_layers=13010)
        X, y = self.get_params(self.master_train)
        out, hn = clf(X, y)

    def lstm(self):
        clf = nn.LSTM(input_size=246, hidden_size=1, num_layers=13010)
        X, y = self.get_params(self.master_train)
        out, hn = clf(X, y)

    def transformer(self):
        clf = nn.Transformer()
        X, y = self.get_params(self.master_train)
        out, hn = clf(X, y)
        pass


TestLearningModels("perc", 10000)