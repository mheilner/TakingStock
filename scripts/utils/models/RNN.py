import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 nonlinearity: str,
                 bias: bool,
                 dropout: float):
        """
        Args:
            input_size (int): Number of expected features per date.
            hidden_size (int): Number of features in each hidden state layer.
            num_layers (int): Number of recurrent layers. Any number greater
                than 1 results in a "stacked RNN".
            nonlinearity (str): Either "tanh" or "relu".
            bias (bool): If bias weights should be used in every layer.
            dropout (float): If non-zero, introduces Dropout layer on the
                outputs of each RNN layer expect the last layer, with dropout
                probability equal to this value.
        """
        super().__init__()

        # Define layers of network
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          nonlinearity=nonlinearity,
                          bias=bias,
                          batch_first=True,
                          dropout=dropout)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=num_layers * hidden_size,
                                out_features=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): A tensor to return predictions for with dimensions
                (batch_size, seq_len, features).

        Returns:
            A torch.Tensor of the predicted next day values with the dimensions
            (batch_size, 1).
        """
        return self.linear(self.flatten(self.rnn(x)[1].transpose_(0,1)))
