import pandas as pd
import torch
from pathlib import Path
from download_data import download_data

# CONSTANTS
DATA_DIR = Path("data/")
DIV_FILE = "dividends.csv"
SPLITS_FILE = "splits.csv"
OHLCV_FILE = "historical.csv"
MON_VAL_FILE = "monthly_valuation_measures_formatted.csv"
QRT_BAL_FILE = "quarterly_balance_sheet_formatted.csv"
QRT_CASH_FILE = "quarterly_cash_flow_formatted.csv"
QRT_FIN_FILE = "quarterly_financials_formatted.csv"

def get_data_tensor(data_dir: str=DATA_DIR,
                    *,
                    div_file: str=DIV_FILE,
                    splits_file: str=SPLITS_FILE,
                    ohlcv_file: str=OHLCV_FILE,
                    mon_val_file: str=MON_VAL_FILE,
                    qrt_bal_file: str=QRT_BAL_FILE,
                    qrt_cash_file: str=QRT_CASH_FILE,
                    qrt_fin_file: str=QRT_FIN_FILE):
    """
    Args:
        data_dir (string): Relative file path to directory with data.
        div_file (string): Filename for dividends CSV or None to not load.
        splits_file (string): Filename for splits CSV or None to not load.
        ohlcv_file (string): Filename for OHLCV CSV or None to not load.
        mon_val_file (string): Filename for valuation CSV or None to not load.
        qrt_bal_file (string): Filename for balance sheet CSV or None to not
            load.
        qrt_cash_file (string): Filename for cash flow CSV or None to not load.
        qrt_fin_file (string): Filename for financials CSV or None to not load.

    Returns:
        A two dimensional torch.Tensor object with the dimensions:
        (date, features).
    """
    # Make data_dir a Path object for convenience
    data_dir = Path(data_dir)

    # Attempt to download freely available CSVs
    download_data(DATA_DIR)

    # Read in each data file. They are kept separate because each has a different
    # format
    dataframes = {}
    if div_file is not None:
        dataframes["dividends"] = pd.read_csv(data_dir / div_file)
    if splits_file is not None:
        dataframes["splits"] = pd.read_csv(data_dir / splits_file)
    if ohlcv_file is not None:
        dataframes["OHLCV"] = pd.read_csv(data_dir / ohlcv_file)
    if mon_val_file is not None:
        dataframes["valuation"] = pd.read_csv(data_dir / mon_val_file)
    if qrt_bal_file is not None:
        dataframes["balance"] = pd.read_csv(data_dir / qrt_bal_file)
    if qrt_cash_file is not None:
        dataframes["cash"] = pd.read_csv(data_dir / qrt_cash_file)
    if qrt_fin_file is not None:
        dataframes["financials"] = pd.read_csv(data_dir / qrt_fin_file)

    # Transpose dataframes that have the date as columns instead of rows
    dataframes["valuation"] = dataframes["valuation"].transpose()
    dataframes["balance"] = dataframes["balance"].transpose()
    dataframes["cash"] = dataframes["cash"].transpose()
    dataframes["financials"] = dataframes["financials"].transpose()

    # TODO: Read in all the dataframes into some standardized, combined tensor

    # TODO: Return this combined tensor
    return None


def get_train_test_sets(train_split: float=0.8, shuffle=True):
    """
    Args:
        train_split (float): What split of the data should be used for training.
        shuffle (boolean): If the train and test tensors should be shuffled
            AFTER the train-test split.

    Returns:
        A tuple where the first element is the training data as a two
        dimensional torch.Tensor object and the second element is the testing
        data as a two dimensional torch.Tensor object, both with the dimensions:
        (date, features).
    """
    # TODO: Return the oldest data as the training data and the newest data as
    #       the testing data
    pass
