import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from os import cpu_count, makedirs, path
from tqdm import tqdm
from .models.Perceptron import Perceptron
from .models.RNN import RNN
from .models.LSTM import LSTM
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
                     model_name: str,
                     train_dataloader: DataLoader,
                     test_dataloader: DataLoader,
                     stopping_lr: float):
        """
        Args:
            opt (torch.optim.Optimizer): The optimizer to use for training.
            model (nn.Module): The model to be trained.
            model_name (str): Name of model for tensorboard reporting.
            train_dataloader (DataLoader): A DataLoader of the training data.
            test_dataloader (DataLoader): A DataLoader of the testing data.
            stopping_lr (float): Instead of training for a number of epochs,
                decrease the learning rate until it is at or below the
                stopping_lr.

        Returns:
            The trained nn.Module instance.
        """
        # Initialize SummaryWriter and metric dict for tensorboard statistics
        makedirs("logs", exist_ok=True)
        writer = SummaryWriter(path.join("logs", model_name))

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

            train_mse = train_loss / len(self.train_dataset)
            train_mae = train_abs_err / len(self.train_dataset)
            print(f"Training MSE Loss: {train_mse}")
            print(f"Training Mean Absolute Error (MAE): {train_mae}")
            writer.add_scalar(tag="train_mse",
                              scalar_value=train_mse,
                              global_step=epoch_num)
            writer.add_scalar(tag="train_mae",
                              scalar_value=train_mae,
                              global_step=epoch_num)

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

            test_mse = test_loss / len(self.test_dataset)
            test_mae = test_abs_err / len(self.test_dataset)
            print(f"Testing MSE Loss: {test_mse}")
            print(f"Testing Mean Absolute Error (MAE): {test_mae}")
            writer.add_scalar(tag="test_mse",
                              scalar_value=test_mse,
                              global_step=epoch_num)
            writer.add_scalar(tag="test_mae",
                              scalar_value=test_mae,
                              global_step=epoch_num)

            # Pass the Learning Rate Scheduler the results
            lr_sched.step(test_loss)

        writer.close()

        return model


    def train_perceptron(self,
                         num_dataloader_processes: int,
                         bias: bool=True,
                         batch_size: int=1,
                         lr: float=0.01,
                         stopping_lr: float=0.0001,
                         use_pretrained: bool=False):
        """
        Args:
            num_dataloader_processes (int): Number of processes to use for
                dataloading.
            bias (bool): If bias weights should be used.
            batch_size (int): How many samples per batch to load.
            lr (float): Initial learning rate.
            stopping_lr (float): Instead of training for a number of epochs,
                decrease the learning rate until it is at or below the
                stopping_lr.
            use_pretrained (bool): If True, use pretrained weights if present
                instead of training a new model.

        Returns:
            Trained instance of Perceptron/Linear model.
        """
        # Return pretrained model if requested and weights are present
        if use_pretrained:
            if path.isfile(path.join("weights", "Perceptron.pt")):
                try:
                    # Create model
                    perceptron_model = Perceptron(
                            input_size=self.train_dataset[0][0].shape[-1],
                            seq_len=self.seq_len,
                            bias=bias)

                    # Load and modify model weights
                    state_dict = torch.load(path.join("weights",
                                                      "Perceptron.pt"))
                    remove_prefix = "_orig_mod."
                    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}

                    # Load weights into model
                    perceptron_model.load_state_dict(state_dict)

                    print("Pretrained weights found!")
                    return perceptron_model
                except Exception as e:
                    print("The error below occured while attempting to " + \
                          f"load model weights. Now training from scratch. {e}")
            else:
                print("Can't find model weights, so training from scratch.")

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
        perceptron_model = torch.compile(Perceptron(
                        input_size=self.train_dataset[0][0].shape[-1],
                        seq_len=self.seq_len,
                        bias=bias)).to(self.device)

        # Create optimizer
        opt = torch.optim.Adam(params=perceptron_model.parameters(), lr=lr)

        # Train model
        perceptron_model = self._train_model(opt=opt,
                                             model=perceptron_model,
                                             model_name="Perceptron",
                                             train_dataloader=train_dataloader,
                                             test_dataloader=test_dataloader,
                                             stopping_lr=stopping_lr)

        # Save weights for the future
        makedirs("weights/", exist_ok=True)
        torch.save(perceptron_model.state_dict(), path.join("weights",
                                                            "Perceptron.pt"))

        return perceptron_model


    def train_RNN(self,
                  num_dataloader_processes: int,
                  hidden_size: int=64,
                  num_layers: int=2,
                  nonlinearity: str="tanh",
                  bias: bool=True,
                  dropout: float=0,
                  batch_size: int=1,
                  lr: float=0.005,
                  stopping_lr: float=0.0001,
                  use_pretrained: bool=False):
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
            use_pretrained (bool): If True, use pretrained weights if present
                instead of training a new model.

        Returns:
            Trained instance of RNN model.
        """
        # Return pretrained model if requested and weights are present
        if use_pretrained:
            if path.isfile(path.join("weights", "RNN.pt")):
                try:
                    # Create model
                    rnn_model = RNN(
                            input_size=self.train_dataset[0][0].shape[-1],
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            nonlinearity=nonlinearity,
                            bias=bias,
                            dropout=dropout)

                    # Load and modify model weights
                    state_dict = torch.load(path.join("weights",
                                                      "RNN.pt"))
                    remove_prefix = "_orig_mod."
                    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}

                    # Load weights into model
                    rnn_model.load_state_dict(state_dict)

                    print("Pretrained weights found!")
                    return rnn_model
                except Exception as e:
                    print("The error below occured while attempting to " + \
                          f"load model weights. Now training from scratch. {e}")
            else:
                print("Can't find model weights, so training from scratch.")

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

        # Train model
        rnn_model = self._train_model(opt=opt,
                                      model=rnn_model,
                                      model_name="RNN",
                                      train_dataloader=train_dataloader,
                                      test_dataloader=test_dataloader,
                                      stopping_lr=stopping_lr)

        # Save weights for the future
        makedirs("weights/", exist_ok=True)
        torch.save(rnn_model.state_dict(), path.join("weights",
                                                     "RNN.pt"))

        return rnn_model


    def train_LSTM(self,
                num_dataloader_processes: int,
                hidden_size: int=50,
                num_layers: int=2,
                lr: float=0.001,
                stopping_lr: float=0.0001,
                batch_size: int=1,
                use_pretrained: bool=False):
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
            use_pretrained (bool): If True, use pretrained weights if present
                instead of training a new model.

        Returns:
            Trained instance of LSTM model.
        """
        # Return pretrained model if requested and weights are present
        if use_pretrained:
            if path.isfile(path.join("weights", "LSTM.pt")):
                try:
                    # Create model
                    lstm_model = LSTM(
                            self.train_dataset[0][0].shape[-1],
                            hidden_size,
                            num_layers)

                    # Load and modify model weights
                    state_dict = torch.load(path.join("weights",
                                                      "LSTM.pt"))
                    remove_prefix = "_orig_mod."
                    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}

                    # Load weights into model
                    lstm_model.load_state_dict(state_dict)

                    print("Pretrained weights found!")
                    return lstm_model
                except Exception as e:
                    print("The error below occured while attempting to " + \
                          f"load model weights. Now training from scratch. {e}")
            else:
                print("Can't find model weights, so training from scratch.")

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
        lstm_model = torch.compile(LSTM(
                        self.train_dataset[0][0].shape[-1],
                        hidden_size,
                        num_layers)).to(self.device)

        # Create optimizer
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)

        # Train model
        lstm_model = self._train_model(opt=optimizer,
                                       model=lstm_model,
                                       model_name="LSTM",
                                       train_dataloader=train_dataloader,
                                       test_dataloader=test_dataloader,
                                       stopping_lr=stopping_lr)

        # Save weights for the future
        makedirs("weights/", exist_ok=True)
        torch.save(lstm_model.state_dict(), path.join("weights",
                                                      "LSTM.pt"))

        return lstm_model


    def train_transformer(self,
                          num_dataloader_processes: int,
                          num_heads: int=6,
                          hidden_size: int=2048,
                          num_layers: int=6,
                          nonlinearity: str="gelu",
                          bias: bool=True,
                          dropout: float=0.1,
                          batch_size: int=1,
                          lr: float=0.005,
                          stopping_lr: float=0.0001,
                          use_pretrained: bool=False):
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
            use_pretrained (bool): If True, use pretrained weights if present
                instead of training a new model.

        Returns:
            Trained instance of Transformer model.
        """
        # Return pretrained model if requested and weights are present
        if use_pretrained:
            if path.isfile(path.join("weights", "Transformer.pt")):
                try:
                    # Create model
                    transformer_model = Transformer(
                            input_size=self.train_dataset[0][0].shape[-1],
                            num_heads=num_heads,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            nonlinearity=nonlinearity,
                            bias=bias,
                            dropout=dropout,
                            seq_len=self.seq_len)

                    # Load and modify model weights
                    state_dict = torch.load(path.join("weights",
                                                      "Transformer.pt"))
                    remove_prefix = "_orig_mod."
                    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}

                    # Load weights into model
                    transformer_model.load_state_dict(state_dict)

                    print("Pretrained weights found!")
                    return transformer_model
                except Exception as e:
                    print("The error below occured while attempting to " + \
                          f"load model weights. Now training from scratch. {e}")
            else:
                print("Can't find model weights, so training from scratch.")

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

        # Train model
        transformer_model = self._train_model(opt=opt,
                                              model=transformer_model,
                                              model_name="Transformer",
                                              train_dataloader=train_dataloader,
                                              test_dataloader=test_dataloader,
                                              stopping_lr=stopping_lr)

        # Save weights for the future
        makedirs("weights/", exist_ok=True)
        torch.save(transformer_model.state_dict(), path.join("weights",
                                                             "Transformer.pt"))

        return transformer_model
