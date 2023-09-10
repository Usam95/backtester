import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(".."))
hist_data_folder = "../../historical_data"


def get_path(ticker):
    symbol_folder = os.path.join(hist_data_folder, ticker)

    # Check if the directory exists
    if not os.path.exists(symbol_folder):
        #logger.error(f"ERROR: Folder for {ticker} does not exist.")
        raise FileNotFoundError(f"Folder for {ticker} does not exist.")

    # Directly check for the file's existence rather than listing all files
    if os.path.isfile(os.path.join(symbol_folder, f"{ticker}.parquet.gzip")):
        return os.path.join(symbol_folder, f"{ticker}.parquet.gzip")
    elif os.path.isfile(os.path.join(symbol_folder, f"{ticker}.csv")):
        return os.path.join(symbol_folder, f"{ticker}.csv")
    else:
        #logger.error(f"ERROR: Could not find any data for {ticker}..")
        raise FileNotFoundError(f"Could not find any data for {ticker}..")


def load_data(ticker):
    data_path = get_path(ticker)

    _, file_extension = os.path.splitext(data_path)
    if file_extension == ".gzip":
        return pd.read_parquet(data_path)
    elif file_extension == ".csv":
        return pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
    else:
        #logger.error(f"ERROR: Unsupported file format: {file_extension} for ticker {ticker}")
        raise ValueError(f"Unsupported file format: {file_extension} for ticker {ticker}")