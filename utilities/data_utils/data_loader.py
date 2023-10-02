import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(".."))
hist_data_folder = "../../historical_data"


def get_path(ticker):
    symbol_folder = os.path.join(hist_data_folder, ticker)

    # Check if the directory exists
    if not os.path.exists(symbol_folder):
        raise FileNotFoundError(f"Folder for {ticker} does not exist.")

    # Directly check for the file's existence rather than listing all files
    if os.path.isfile(os.path.join(symbol_folder, f"{ticker}.parquet.gzip")):
        return os.path.join(symbol_folder, f"{ticker}.parquet.gzip")
    elif os.path.isfile(os.path.join(symbol_folder, f"{ticker}.csv")):
        return os.path.join(symbol_folder, f"{ticker}.csv")
    else:
        raise FileNotFoundError(f"Could not find any data for {ticker}..")


def load_data(config):
    symbol = config["dataset_conf"]["symbol"]
    start_date = config["dataset_conf"]["start_date"]
    end_date = config["dataset_conf"]["end_date"]

    data_path = get_path(symbol)

    _, file_extension = os.path.splitext(data_path)
    if file_extension == ".gzip":
        data = pd.read_parquet(data_path)
    elif file_extension == ".csv":
        data = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
    else:
        raise ValueError(f"Unsupported file format: {file_extension} for symbol {symbol}")

    # Adjusting start and end dates
    if start_date == "" or start_date not in data.index:
        start_date = data.index[0].strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Start date not provided or not in data. Using first available date: {start_date}")

    if end_date == "" or end_date not in data.index:
        end_date = data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"End date not provided or not in data. Using last available date: {end_date}")

    data = data.loc[start_date:end_date]

    return data