import torch
from torch import nn

class Transformer(nn.Module):
    def __init__(self,
                 input_size: int,
                 num_heads: int,
                 hidden_size: int,
                 num_layers: int,
                 nonlinearity: str,
                 bias: bool,
                 dropout: float,
                 seq_len: int):
        """
        Args:
            input_size (int): Number of expected features per date (embedding
                size).
            num_heads (int): Number of heads used in Multi-Headed Attention.
            hidden_size (int): Number of features in each FFN hidden state layer.
            num_layers (int): Number of decoders to stack.
            nonlinearity (str): Either "gelu" or "relu".
            bias (bool): If bias weights should be used in each FFN and
                LayerNorm layer.
            dropout (float): If non-zero, introduces Dropout layers with dropout
                probability equal to this value.
            seq_len (int): Target length (context window size) for decoder.
        """
        super().__init__()

        self.memory = nn.Parameter(torch.randn(seq_len, input_size))

        # Define layers of network
        decoder_layer = nn.TransformerDecoderLayer(d_model=input_size,
                                                   nhead=num_heads,
                                                   dim_feedforward=hidden_size,
                                                   dropout=dropout,
                                                   activation=nonlinearity,
                                                   batch_first=True,
                                                   bias=bias)
        self.decoder_stack = nn.TransformerDecoder(decoder_layer=decoder_layer,
                                                   num_layers=num_layers)
        self.FFN = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=hidden_size),
                nn.GELU(),
                nn.Linear(in_features=hidden_size, out_features=1))

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

        # Run input through Transformer decoder stack
        x = self.decoder_stack(tgt=x, memory=torch.stack([self.memory] * x.shape[0]))

        # Run last predicted token through FFN
        return self.FFN(x[:, -1, :])
