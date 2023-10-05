import os
import pandas as pd
import tqdm
import pickle
import sys
import time
import datetime
from credentials import *
from logger import Logger
from binance.client import Client

from data_loader import load_data, get_path

class DataRetriever:

    """ Class for retrieving historical data from Binanace for cryptocurrency coins.
        For each coin it creates a separate folder in hist_data directory and stores there the downloaded data.
        Requires the binance public and secret API keys to be provided in credentials.py modul"


    Attributes
    ============
    folder: str
        directory where the downloaded data are stored

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
        self.hist_data_folder = folder
        self.logger = self.logger = Logger().get_logger()
        self.client = None
        self.symbol_df = None
        self.symbols = None

    def connect(self): 
        self.client = Client(api_key = testnet_api_key, api_secret = testnet_secret_key, tld = "com", testnet=True)

        if len(self.client.ping()) == 0:
            self.logger.info(f"Connection to server is established.")
        else:
            self.logger.error(f"Could not connect to server. Exit.")
            exit(1)
            
    def get_all_tickers(self): 
        self.symbols = self.client.get_all_tickers()
        self.symbol_df = pd.DataFrame(self.symbols)
        self.symbol_df.reset_index(inplace=True, drop=True)
        self.symbols = self.symbol_df.symbol.to_list()
    
    def get_only_tickers_containing_symbol(self, symbol="USDT"):
        self.symbols = self.client.get_all_tickers()
        self.symbol_df = pd.DataFrame(self.symbols)
        self.symbol_df = self.symbol_df[self.symbol_df.symbol.str.contains(symbol)]
        self.symbol_df.reset_index(inplace=True, drop=True)
        
        self.symbols = self.symbol_df.symbol.to_list()
       
    def create_ticker_folder(self, ticker):
        path = os.path.join(self.folder, ticker)

        if not os.path.exists(path):
            self.logger.info(f"Creating folder for {ticker} to store historical data.")
            os.mkdir(path)                                       
                                           
    def load_tickers(self, filename="symbols.txt"):
        with open(filename, "rb") as fp:   # Unpickling
            self.symbols = pickle.load(fp)

    def remove_ticker_and_store_list(self, ticker, filename="symbols.txt"): 
        try: 
            self.symbols.remove(ticker)
        except ValueError: 
            self.logger.error(f"{ticker} is not is the list of defined symbols.")

        with open("symbols.txt", "wb") as fp:   # Pickling
            pickle.dump(self.symbols, fp)
            self.logger.into(f"Stored the list of symbols to be updated in the file {filename}.")

    def get_ticker_historical_data(self, ticker, interval, start_t=None):
        max_attempts = 12
        attempt = 0

        while attempt < max_attempts:
            try:
                now = datetime.datetime.now()
                end_str = now.strftime("%Y-%m-%d")

                # Ensure start_t is in string format
                if isinstance(start_t, (datetime.date, datetime.datetime)):
                    start_str = start_t.strftime("%Y-%m-%d")
                else:
                    start_str = start_t
                self.logger.info(f"{ticker=}, {interval=}, {start_str=}, {end_str=}")
                bars = self.client.get_historical_klines(symbol=ticker, interval=interval, start_str=start_str,
                                                         end_str=end_str)
            except Exception as e:
                self.logger.error(f"Could not retrieve data for {ticker}..")
                self.logger.error(f"Error message: {e}")
                self.logger.error("Trying to reconnect to server..")

                # Increment attempt
                attempt += 1
                self.logger.info(f"Attempt: {attempt}")

                # Try to reconnect (assuming it was a connection error).
                self.connect()

                # wait 5 seconds before trying again
                time.sleep(5)
            else:
                df = pd.DataFrame(bars, columns=["Open Time", "Open", "High", "Low", "Close", "Volume",
                                                 "Close Time", "Quote Asset Volume", "Number of Trades",
                                                 "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume",
                                                 "Ignore"])

                df["Date"] = pd.to_datetime(df["Open Time"], unit="ms")
                df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

                return df

        # If loop completes and max_attempts is reached
        self.logger.error(f"Max attempts for reconnections achieved.. Exiting..")
        exit(1)

    def store_df(self, df, path, parquet=True):
        try:
            if parquet:
                # Convert relevant columns from string to float
                df['Open'] = df['Open'].astype(float)
                df['High'] = df['High'].astype(float)
                df['Low'] = df['Low'].astype(float)
                df['Close'] = df['Close'].astype(float)
                df['Volume'] = df['Volume'].astype(float)
                df.to_parquet(path, compression='gzip', engine='pyarrow')
            else:
                # Convert timestamp to datetime and set as inde
                symbol = os.path.basename(path)
                path = os.path.join(path, f"{symbol}.csv")
                self.logger.info(f"Storing data for {symbol} into the file {path}.")
                df.to_csv(path)

        except Exception as e:
            self.logger.error(f"Error occurred while storing the DataFrame: {e}")

        except Exception as e:
            self.logger.error(f"Error occurred while storing the DataFrame: {e}")

    def update_historical_data_for_symbol(self, symbol, interval="1m"):
        # Step 1: Read the existing data
        df = load_data(symbol)

        # Step 2: Identify the last date
        last_date = df.index[-1].date()

        # Check if update is needed
        today = datetime.datetime.now().date()
        if last_date >= today:
            self.logger.info(f"Data for {symbol} with granularity {interval} is up-to-date.")
            return

        # Step 3: Fetch and append new data
        start_date = last_date + datetime.timedelta(days=1)
        new_data = self.get_ticker_historical_data(symbol, interval=interval, start_t=start_date)

        if new_data is not None and not new_data.empty:
            # Convert to desired format
            new_data["Date"] = pd.to_datetime(new_data.iloc[:, 0], unit="ms")
            self.logger.info(new_data.columns)
            new_data = new_data[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
            new_data.set_index("Date", inplace=True)

            # Merge the old and new data
            df = pd.concat([df, new_data])

            # Store the updated data
            path = get_path(symbol)
            self.store_df(df, path)
            self.logger.info(f"Data for {symbol} with granularity {interval} updated successfully.")
        else:
            self.logger.info(f"No new data found for {symbol} with granularity {interval}.")

    def retrieve_all_historical_data(self):
        total_start_t = time.time()

        for ticker in tqdm.tqdm(self.symbols):
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

    symbols = ["XRPUSDT", "BTCUSDT", "TRXUSDT", "LTCUSDT"]
    dataRetriever = DataRetriever()
    dataRetriever.connect()

    for symbol in symbols:
        dataRetriever.update_historical_data_for_symbol(symbol)

