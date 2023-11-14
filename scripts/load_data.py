import pandas as pd
import numpy as np
import torch
from utils.StockDataset import StockDataset
from pathlib import Path
from download_data import download_data
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype

# CONSTANTS
DATA_DIR = Path(__file__).parent / "data"
DIV_FILE = "dividends.csv"
SPLITS_FILE = "splits.csv"
OHLCV_FILE = "historical.csv"
MON_VAL_FILE = "monthly_valuation_measures_formatted.csv"
QRT_BAL_FILE = "quarterly_balance_sheet_formatted.csv"
QRT_CASH_FILE = "quarterly_cash_flow_formatted.csv"
QRT_FIN_FILE = "quarterly_financials_formatted.csv"

def get_data_tensor(relative_change: bool=True,
                    target_column: str="Close",
                    data_dir: str=DATA_DIR,
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
        relative_change (bool): If True, inputs will be fraction of change from
            previous day instead of absolute values.
        target_column (string): Name of column from one of the CSVs to be the
            the first feature.
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
        (date, features). Note tha the first feature will be the target_column.
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
    dataframes_to_transpose = ["valuation", "balance", "cash", "financials"]
    for df_key in dataframes_to_transpose:
        dataframes[df_key] = dataframes[df_key].transpose()

    # Set first row as header when column names are rows of data
    dataframes_fix_headers = ["valuation", "balance", "cash", "financials"]
    for df_key in dataframes_fix_headers:
        dataframes[df_key].columns = dataframes[df_key].iloc[0]
        dataframes[df_key] = dataframes[df_key][1:]

    # Make index "name" into "Date" column, which is needed after transposing
    dataframes_fix_date_col = ["valuation", "balance", "cash", "financials"]
    for df_key in dataframes_fix_date_col:
        # Make index the first column
        dataframes[df_key].reset_index(inplace=True)
        
        # Name first column "Date"
        dataframes[df_key].rename(columns={"index": "Date"}, inplace=True)

    # Remove any rows that aren't for a date but are instead the TTM.
    # NOTE: TTM = The trailing 12-month. Refers to past 12 months of data.
    for df_key in dataframes:
        dataframes[df_key] = dataframes[df_key][dataframes[df_key].Date != "ttm"]

    # Standardize all Date columns to datetime64 objects to enable merging
    for df_key in dataframes:
        dataframes[df_key].Date = pd.to_datetime(dataframes[df_key].Date)

    # Assert that there aren't any illegal or blank entries in every Date column
    for df_key in dataframes:
        assert is_datetime64_any_dtype(dataframes[df_key].Date) \
                and dataframes[df_key].Date.isna().sum() == 0, \
                f"Not all dates in the '{df_key}' DataFrame could be " \
                "converted to a datetime object! Check for weird entries " \
                "amongst dates in the respective CSV file! The following " \
                "might be a good place to start: \n" \
                f"{dataframes[df_key][dataframes[df_key].Date.isna()]}"

    # Merge all dataframes into one master dataframe
    master_df = None
    for dataframe in dataframes.values():
        if master_df is None:
            master_df = dataframe
        else:
            master_df = master_df.merge(dataframe,
                                        how="outer",
                                        on="Date",
                                        sort=False)
    assert master_df is not None, "No data to turn into a torch.Tensor!"

    # Sort master_df so that dates are in ascending order
    master_df.sort_values("Date", inplace=True, ignore_index=True)

    # Drop the Date column, which is no longer needed after merging and sorting
    master_df.drop(columns="Date", inplace=True)

    # Convert "Stock Splits" entries from strings (ex: "2:1") to floats
    if "Stock Splits" in master_df.columns:
        # Get all stock splits
        stock_splits = set(master_df["Stock Splits"].dropna().unique())

        # Convert stock splits into floats
        conv_dict = {}
        for ss in stock_splits:
            new_qty, old_qty = ss.split(":")
            ratio = float(new_qty) / float(old_qty)
            conv_dict[ss] = ratio

        # Replace stock splits with floats
        master_df.replace(to_replace=conv_dict, inplace=True)

    # Convert entire dataframe into floats and drop Date column
    for col_name in master_df.columns:
        if is_string_dtype(master_df[col_name].dtype):
            # A lot of the strings have commas in them, which have to be removed
            master_df[col_name] = master_df[col_name].str\
                                                     .replace(",", "")\
                                                     .astype(float)

    # Fill in NA/NaN values where we don't have data.
    # NOTE: We forward fill first because in real life, we won't have future
    # data. The back fill is to give us values at the beginning of the data to
    # prevent overfit (ex: guessing low because there's a NA/NaN value, meaning
    # we're at the beginning of the stock's history).
    master_df = master_df.ffill(axis=0)
    master_df = master_df.bfill(axis=0)

    # Reorder the columns so that target_column is first
    assert target_column in master_df.columns, \
            f"'{target_column}' is not one of the columns from a CSV file!"
    
    new_col_order = list(master_df.columns)
    new_col_order.remove(target_column)
    master_df = master_df[[target_column] + new_col_order]

    # Convert master_df to a standardized, combined PyTorch tensor
    master_tens = torch.from_numpy(master_df.to_numpy(dtype=np.float32))

    # If relative_change, convert data to fraction of change from previous day
    if relative_change:
        master_tens_rd = master_tens.roll(1, 0)
        master_tens = (master_tens - master_tens_rd) / torch.abs(master_tens_rd)

        # Discard first day because roll goes over edge of dataset
        master_tens = master_tens[1:, :]

    return master_tens


def get_train_test_datasets(data_tensor: torch.Tensor,
                            seq_len: int,
                            train_split: float=0.8):
    """
    Args:
        data_tensor (torch.Tensor): 2D tensor of dimensions (date, features)
            where the first feature is the target feature.
        seq_len (int): How many days of data to return before the target.
        train_split (float): What split of the data should be used for training.

    Returns:
        A tuple where the first element is the training dataset and the second
        element is the testing dataset, both objects inheriting from the class
        torch.utils.data.Dataset.
    """
    train_tensor = data_tensor[:int(data_tensor.shape[0] * train_split)]
    test_tensor = data_tensor[int(data_tensor.shape[0] * train_split):]

    return (StockDataset(train_tensor, seq_len),
            StockDataset(test_tensor, seq_len))
