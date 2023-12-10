import torch
from torch import nn

class Perceptron(nn.Module):
    def __init__(self,
                 input_size: int,
                 seq_len: int,
                 bias: bool):
        """
        Args:
            input_size (int): Number of expected features per date.
            bias (bool): If bias weights should be used in every layer.
        """
        super().__init__()

        # Define layers of network
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=input_size * seq_len,
                                out_features=1,
                                bias=bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): A tensor to return predictions for with dimensions
                (batch_size, seq_len, features).

        Returns:
            A torch.Tensor of the predicted next day values with the dimensions
            (batch_size, 1).
        """
        # Run input through norm layer to prevent nan values from outliers later
        x = nn.functional.normalize(x, dim=2)

        return self.linear(self.flatten(x))
