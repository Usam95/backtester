import os
import pandas as pd
import tqdm
import pickle
import sys
import time
from datetime import datetime, timedelta
from credentials import *
from logger import Logger
from binance.client import Client


class DataRetriever:

    """ Class for retrieving historical data from Binanace for cryptocurrency coins.
        For each coin it creates a separate folder in hist_data directory and stores there the downloaded data.
        Requires the binance public and secret API keys to be provided in credentials.py modul"


    Attributes
    ============
    folder: str
        directory where the downloaded data are stored

    #start: str
        start date for data import
    #end: str
        end date for data import


    Methods
    =======
    connect:
        connects to binance server.

    test_strategy:
        prepares the data and backtests the trading strategy incl. reporting (wrapper).

    get_all_tickers:
        get all tradable tickers by binance .

    get_only_tickers_containing_symbol:
        get all tradable tickers containing given symbol (USDT by default)

    create_ticker_folder:
        used to create a separate folder for each ticker in "hist_data" dictionary

    store_df:
        stores the downloaded historical data as ".csv" file in corresponding folder.

    get_ticker_historical_data:
        download historical data for a given ticker.

    retrieve_all_historical_data:
        download and store historical data for all tradable (filtered out) tickers.

    """

    def __init__(self, folder="../hist_data"):
        self.folder = folder
        self.logger = self.logger = Logger().get_logger()
        self.client = None

    def connect(self): 
        self.client = Client(api_key = testnet_api_key, api_secret = testnet_secret_key, tld = "com", testnet=True)

        if len(self.client.ping()) == 0:
            self.logger.info(f"Connection to server is established.")
        else:
            self.logger.error(f"Could not connect to server. Exit.")
            exit(1)
            
    def get_all_tickers(self): 
        self.tickers = self.client.get_all_tickers()
        self.tickers_df = pd.DataFrame(self.tickers)
        self.tickers_df.reset_index(inplace=True, drop=True)   
        self.tickers = self.tickers_df.symbol.to_list()
    
    def get_only_tickers_containing_symbol(self, symbol="USDT"):
        self.tickers = self.client.get_all_tickers()                 
        self.tickers_df = pd.DataFrame(self.tickers)
        self.tickers_df = self.tickers_df[self.tickers_df.symbol.str.contains(symbol)]
        self.tickers_df.reset_index(inplace=True, drop=True)   
        
        self.tickers = self.tickers_df.symbol.to_list()
       
    def create_ticker_folder(self, ticker):
        path = os.path.join(self.folder, ticker)

        if not os.path.exists(path):
            self.logger.info(f"Creating folder for {ticker} to store historical data.")
            os.mkdir(path)                                       
                                           
    def load_tickers(self, filename="symbols.txt"):
        with open(filename, "rb") as fp:   # Unpickling
            self.tickers = pickle.load(fp)

    def remove_ticker_and_store_list(self, ticker, filename="symbols.txt"): 
        try: 
            self.tickers.remove(ticker)
        except ValueError: 
            self.logger.error(f"{ticker} is not is the list of defined symbols.")

        with open("symbols.txt", "wb") as fp:   # Pickling
            pickle.dump(self.tickers, fp)
            self.logger.into(f"Stored the list of symbols to be updated in the file {filename}.")

    def store_df(self, df, path):
        try:
            # Convert timestamp to datetime and set as index
            df["Date"] = pd.to_datetime(df.iloc[:, 0], unit="ms")
            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            df.set_index("Date", inplace=True)

            symbol = os.path.basename(path)
            path = os.path.join(path, f"{symbol}.csv")
            self.logger.info(f"Storing data for {symbol} into the file {path}.")
            df.to_csv(path)

        except Exception as e:
            self.logger.error(f"Error occurred while storing the DataFrame: {e}")

        except Exception as e:
            self.logger.error(f"Error occurred while storing the DataFrame: {e}")

    def get_ticker_historical_data(self, ticker, interval, start_t=None):
        max_attempts = 12
        attempt = 0
        while True: 
            try: 

                now = datetime.now()
                end_str = now.strftime("%Y-%m-%d")
                bars = self.client.get_historical_klines(symbol=ticker, interval=interval, start_str=start_t, end_str=end_str)
            except Exception as e:
                self.logger.error(f"Could not retrieve data for {ticker}..", end = " | ")
                self.logger.error(f"Error message: {e}")
                self.logger.error("Trying to reconnect to server..")
                
            else: 
                df = pd.DataFrame(bars)
                return df 
            
            finally: 
                # wait 5 seconds before trying to reconnect
                time.sleep(5)
                attempt += 1
                self.logger.info(f"Attempt: {attempt}")
                # Try to reconnect (assuming it was the connection error).
                self.connect()
                
                # stop the programm if max attempts reached
                if attempt >= max_attempts: 
                    self.logger.error(f"Max attempts for reconnections achieved.. Exiting..")
                    exit(1)

    def get_path(self, ticker):
        symbol_folder = os.path.join(self.hist_data_folder, ticker)

        # Check if the directory exists
        if not os.path.exists(symbol_folder):
            self.logger.error(f"ERROR: Folder for {ticker} does not exist.")
            raise FileNotFoundError(f"Folder for {ticker} does not exist.")

        # Directly check for the file's existence rather than listing all files
        if os.path.isfile(os.path.join(symbol_folder, f"{ticker}.parquet.gzip")):
            return os.path.join(symbol_folder, f"{ticker}.parquet.gzip")
        elif os.path.isfile(os.path.join(symbol_folder, f"{ticker}.csv")):
            return os.path.join(symbol_folder, f"{ticker}.csv")
        else:
            self.logger.error(f"ERROR: Could not find any data for {ticker}..")
            raise FileNotFoundError(f"Could not find any data for {ticker}..")

    def load_data(self, ticker):
        data_path = self.get_path(ticker)

        _, file_extension = os.path.splitext(data_path)
        if file_extension == ".gzip":
            return pd.read_parquet(data_path)
        elif file_extension == ".csv":
            return pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
        else:
            self.logger.error(f"ERROR: Unsupported file format: {file_extension} for ticker {ticker}")
            raise ValueError(f"Unsupported file format: {file_extension} for ticker {ticker}")

    def update_historical_data_for_symbol(self, symbol, interval="1m"):
        # Step 1: Read the existing data
        df = self.load_data(symbol)

        # Step 2: Identify the last date
        last_date = df.index[-1].date()

        # Check if update is needed
        today = datetime.now().date()
        if last_date >= today:
            self.logger.info(f"Data for {symbol} with granularity {interval} is up-to-date.")
            return

        # Step 3: Fetch and append new data
        start_date = last_date + timedelta(days=1)
        new_data = self.get_ticker_historical_data(symbol, interval=interval, start_t=start_date)

        if new_data is not None and not new_data.empty:
            # Convert to desired format
            new_data["Date"] = pd.to_datetime(new_data.iloc[:, 0], unit="ms")
            new_data = new_data[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            new_data.set_index("Date", inplace=True)

            # Merge the old and new data
            df = pd.concat([df, new_data])

            # Store the updated data
            path = self.get_path(symbol)
            df.to_csv(path)

            self.logger.info(f"Data for {symbol} with granularity {interval} updated successfully.")
        else:
            self.logger.info(f"No new data found for {symbol} with granularity {interval}.")

    def retrieve_all_historical_data(self):
        total_start_t = time.time()

        for ticker in tqdm.tqdm(self.tickers):
            try:
                start_t = time.time()

                # Use the update function directly
                self.update_historical_data_for_symbol(ticker)

                end_t = time.time()
                self.logger.info(f"Downloading/ Updating data for {ticker} took {(end_t - start_t) / 60} mins..")
                print("=" * 80)
                time.sleep(1)

            except Exception as e:
                self.logger.error(f"Error {e} occurred trying to update hist data for ticker {ticker}..")
                continue

        total_end_t = time.time()
        self.logger.info(
            f"Downloading/ Updating all ticker historical data took in total {(total_end_t - total_start_t) / 60} mins..")


if __name__ == "__main__":
    symbols = ["ADAUSDT", "SOLUSDT", "DOTUSDT", "XRPUSDT"]
    dataRetriever = DataRetriever()
    dataRetriever.connect()
    dataRetriever.tickers = symbols
    dataRetriever.retrieve_all_historical_data()
