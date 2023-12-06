import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        """
        LSTM Model for time series prediction.

        Args:
            input_size (int): Number of expected features in the input `x`.
            hidden_size (int): Number of features in the hidden state `h`.
            num_layers (int): Number of recurrent layers in the LSTM.
            output_size (int, optional): Number of features in the output. Default is 1.

        The LSTM model is designed for time series prediction tasks. It takes a sequence
        of data points as input and predicts the next value in the sequence.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer to output the prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor with dimensions (batch_size, seq_len, features).

        Returns:
            torch.Tensor: Output tensor with predictions, dimensions (batch_size, output_size).
        """
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))

        # Pass the output of the last time step to the fully connected layer
        out = self.fc(out[:, -1, :])
        return out