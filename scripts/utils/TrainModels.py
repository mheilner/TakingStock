import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from os import cpu_count
from tqdm import tqdm
from sklearn.linear_model import Perceptron
from .models.RNN import RNN
from .models.LSTM import LSTMModel
from .models.Transformer import Transformer
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

        # Notice we're actually using just Squared Error (squared L2) loss here
        self.loss_fn = nn.MSELoss(reduction="sum")

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

    def _train_model(self,
                     opt: torch.optim.Optimizer,
                     model: nn.Module,
                     train_dataloader: DataLoader,
                     test_dataloader: DataLoader,
                     stopping_lr: float):
        """
        Args:
            opt (torch.optim.Optimizer): The optimizer to use for training.
            model (nn.Module): The model to be trained.
            train_dataloader (DataLoader): A DataLoader of the training data.
            test_dataloader (DataLoader): A DataLoader of the testing data.
            stopping_lr (float): Instead of training for a number of epochs,
                decrease the learning rate until it is at or below the
                stopping_lr.

        Returns:
            The trained nn.Module instance.
        """
        # Create Learning Rate Scheduler
        lr_sched = ReduceLROnPlateau(opt, patience=5, verbose=True)

        # Train until lr_sched and the lack of improvement on the test dataset
        # breaks us out of the training loop
        epoch_num = 0
        while opt.param_groups[0]["lr"] > stopping_lr:
            epoch_num += 1
            print(f"\nEpoch #{epoch_num}:")

            # Training loop
            train_abs_err = 0
            train_loss = 0
            for batch_feats, batch_lbls in tqdm(train_dataloader):
                # Move the data to the device for training, either GPU or CPU
                batch_feats = batch_feats.to(device=self.device,
                                             non_blocking=True)
                batch_lbls = batch_lbls.to(device=self.device,
                                           non_blocking=True)

                # Reset gradients for optimizer
                opt.zero_grad()

                # Get model predictions
                preds = model(batch_feats)

                # Calculate absolute error between preds and labels
                train_abs_err += torch.sum(torch.abs(preds - batch_lbls))

                # Calculate loss (multiply by 10 because losses are small)
                loss = self.loss_fn(preds, batch_lbls) * 10
                train_loss += loss

                # Backprop with gradient clipping to prevent exploding gradients
                loss.backward()
                opt.step()

            print(f"Training MSE Loss: {train_loss / len(self.train_dataset)}")
            print("Training Mean Absolute Error (MAE): "
                  f"{train_abs_err / len(self.train_dataset)}")

            # Testing loop
            test_abs_err = 0
            test_loss = 0
            with torch.inference_mode():
                for batch_feats, batch_lbls in tqdm(test_dataloader):
                    # Move the data to the device for testing, either GPU or CPU
                    batch_feats = batch_feats.to(device=self.device,
                                                 non_blocking=True)
                    batch_lbls = batch_lbls.to(device=self.device,
                                               non_blocking=True)

                    # Get model predictions
                    preds = model(batch_feats)

                    # Calculate absolute error between preds and labels
                    test_abs_err += torch.sum(torch.abs(preds - batch_lbls))

                    # Calculate loss (multiply by 10 because losses are small)
                    loss = self.loss_fn(preds, batch_lbls) * 10
                    test_loss += loss

            print(f"Testing MSE Loss: {test_loss / len(self.test_dataset)}")
            print("Testing Mean Absolute Error (MAE): "
                  f"{test_abs_err / len(self.test_dataset)}")

            # Pass the Learning Rate Scheduler the results
            lr_sched.step(test_loss)

        return model

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
                  num_dataloader_processes: int,
                  hidden_size: int=64,
                  num_layers: int=2,
                  nonlinearity: str="tanh",
                  bias: bool=True,
                  dropout: float=0,
                  batch_size: int=1,
                  lr: float=0.005,
                  stopping_lr: float=0.0001):
        """
        Args:
            num_dataloader_processes (int): Number of processes to use for
                dataloading.
            hidden_size (int): Number of features in each hidden state layer.
            num_layers (int): Number of recurrent layers. Any number greater
                than 1 results in a "stacked RNN".
            nonlinearity (str): Either "tanh" or "relu".
            bias (bool): If bias weights should be used in every layer.
            dropout (float): If non-zero, introduces Dropout layer on the
                outputs of each RNN layer expect the last layer, with dropout
                probability equal to this value.
            batch_size (int): How many samples per batch to load.
            lr (float): Initial learning rate.
            stopping_lr (float): Instead of training for a number of epochs,
                decrease the learning rate until it is at or below the
                stopping_lr.

        Returns:
            Trained instance of RNN model.
        """
        # Get train and test dataloaders
        train_dataloader = DataLoader(dataset=self.train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_dataloader_processes)
        test_dataloader = DataLoader(dataset=self.test_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_dataloader_processes)

        # Create RNN instance
        rnn_model = torch.compile(RNN(
                        input_size=self.train_dataset[0][0].shape[-1],
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        nonlinearity=nonlinearity,
                        bias=bias,
                        dropout=dropout)).to(self.device)

        # Create optimizer
        opt = torch.optim.Adam(params=rnn_model.parameters(), lr=lr)

        return self._train_model(opt=opt,
                                 model=rnn_model,
                                 train_dataloader=train_dataloader,
                                 test_dataloader=test_dataloader,
                                 stopping_lr=stopping_lr)

    def train_LSTM(self,
                num_dataloader_processes: int,
                hidden_size: int = 50,
                num_layers: int = 2,
                lr: float = 0.001,
                stopping_lr: float = 0.0001,
                batch_size: int = 1):
        """
        Trains an LSTM model on the provided dataset.

        Args:
            num_dataloader_processes (int): Number of processes to use for
                dataloading.
            hidden_size (int): Number of features in the hidden state `h`.
            num_layers (int): Number of recurrent layers in the LSTM.
            lr (float): Learning rate for the optimizer.
            stopping_lr (float): Learning rate at which training stops.
            batch_size (int): Batch size for training and testing data.

        Returns:
            Trained instance of LSTM model.
        """
        # Get train and test dataloaders
        train_dataloader = DataLoader(dataset=self.train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_dataloader_processes)
        test_dataloader = DataLoader(dataset=self.test_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_dataloader_processes)
        
        # Create LSTM model instance
        lstm_model = LSTMModel(self.train_dataset[0][0].shape[-1], hidden_size, num_layers).to(self.device)

        # Create optimizer
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)

        # Train the model using the _train_model method
        return self._train_model(optimizer, lstm_model, train_dataloader, test_dataloader, stopping_lr)


    def train_transformer(self,
                          num_dataloader_processes: int,
                          num_heads: int=6,
                          hidden_size: int=2048,
                          num_layers: int=6,
                          nonlinearity: str="gelu",
                          bias: bool=True,
                          dropout: float=0.1,
                          batch_size: int=32,
                          lr: float=0.005,
                          stopping_lr: float=0.0001):
        """
        Args:
            num_dataloader_processes (int): Number of processes to use for
                dataloading.
            num_heads (int): Number of heads used in Multi-Headed Attention.
            hidden_size (int): Number of features in each FFN hidden state layer.
            num_layers (int): Number of decoders to stack.
            nonlinearity (str): Either "gelu" or "relu".
            bias (bool): If bias weights should be used in each FFN and
                LayerNorm layer.
            dropout (float): If non-zero, introduces Dropout layers with dropout
                probability equal to this value.
            batch_size (int): How many samples per batch to load.
            lr (float): Initial learning rate.
            stopping_lr (float): Instead of training for a number of epochs,
                decrease the learning rate until it is at or below the
                stopping_lr.

        Returns:
            Trained instance of Transformer model.
        """
        # Get train and test dataloaders
        train_dataloader = DataLoader(dataset=self.train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_dataloader_processes)
        test_dataloader = DataLoader(dataset=self.test_dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=num_dataloader_processes)

        # Create Transformer instance
        transformer_model = torch.compile(Transformer(
                        input_size=self.train_dataset[0][0].shape[-1],
                        num_heads=num_heads,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        nonlinearity=nonlinearity,
                        bias=bias,
                        dropout=dropout,
                        seq_len=self.seq_len)).to(self.device)

        # Create optimizer
        opt = torch.optim.Adam(params=transformer_model.parameters(), lr=lr)

        return self._train_model(opt=opt,
                                 model=transformer_model,
                                 train_dataloader=train_dataloader,
                                 test_dataloader=test_dataloader,
                                 stopping_lr=stopping_lr)
