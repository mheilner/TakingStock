from torch.utils.data import Dataset
import torch

class StockDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor, seq_len: int):
        """
        Args:
            data_tensor (torch.Tensor): Tensor of dimensions (date, features)
                where the first feature is the target.
            seq_len (int): How many days of data to return before the target.
        """
        assert data_tensor.shape[0] > seq_len, \
                "There are less days in the data_tensor "\
                f"({data_tensor.shape[0]}) than the sequence length " \
                f"({seq_len}) and the following day!"

        self.data_tensor = data_tensor
        self.seq_len = seq_len

    def __len__(self):
        return self.data_tensor.shape[0] - self.seq_len

    def __getitem__(self, index: int):
        # Skip the first column of data, seeing that it's the target feature
        return (self.data_tensor[index:index + self.seq_len, 1:],
                self.data_tensor[index + self.seq_len, 0])
