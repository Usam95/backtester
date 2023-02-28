import os 
import pandas as pd
import numpy as np
import tqdm
import pickle
import sys
import time
from datetime import datetime

sys.path.append("")
sys.path.append("../binance/")

from credentials import *
from binance.client import Client


class DataRetriever: 
    
    ''' Class for retrieving historical data from Binanace for cryptocurrency coins.
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
        
    ''' 
    def __init__(self, folder="../hist_data"):
        self.folder = folder

    def connect(self): 
        self.client = Client(api_key = testnet_api_key, api_secret = testnet_secret_key, tld = "com", testnet=True)

        if len(self.client.ping()) == 0:
            print(f"Connection to server is established.")
        else: 
            print(f"Could not connect to server.")
            exit(0)
            
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
            print(f"Creating folder for {ticker}")
            os.mkdir(path)                                       
                                           
    def load_tickers(self, filename="symbols.txt"):
        with open(filename, "rb") as fp:   # Unpickling
            self.tickers = pickle.load(fp)

    def remove_ticker_and_store_list(self, ticker, filename="symbols.txt"): 
        try: 
            self.tickers.remove(ticker)
        except ValueError: 
            print(f"{ticker} is not is the symbols list.")

        with open("symbols.txt", "wb") as fp:   # Pickling
            pickle.dump(self.tickers, fp)
            print(f"Stored the list of tickers in {filename}")
   
    def store_df(self, df, path):
        df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
        df.columns = ["Open Time", "Open", "High", "Low", "Close",
                  "Volume", "Clos Time", "Quote Asset Volume", 
                  "Number of Trades", "Taker Buy Base Asset Volume",
                  "Taker Buy Quote Asset Volume", "Ignore", "Date" ]
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.set_index("Date", inplace = True)

        #for column in df.columns:
        #    df[column] = pd.to_numeric(df[column], errors = "coerce")

        symbol = path.split("/")[-1]
        path = path + "/" + symbol + ".csv"
        print(f"Storing data for {symbol} as {path}")
        df.to_csv(path)                                       
        
    def get_ticker_historical_data(self, ticker, start_t=None):
        max_attempts = 12
        attempt = 0
        success = False
        #timestamp = "2022-01-03"
        while True: 
            try: 
                print(f"Retrieving data for {ticker} with 1m interval..")
                #timestamp = self.client._get_earliest_valid_timestamp(symbol = ticker, interval = "1m")
                if start_t is not None:
                    start_t = start_t
                else:
                    start_t = pd.to_datetime(timestamp, unit = "ms") # earliest data available on Binance
                #timestamp = "2022-10-08"
                now = datetime.now()
                end_str = now.strftime("%Y-%m-%d")
                print(f"Data for {ticker} are available from {start_t}..")
                print(f"{end_str=}")
                print(f"Loading bars..")
                bars = self.client.get_historical_klines(symbol = ticker, interval = "1m", start_str = start_t, end_str=end_str)
                print(f"Loaded bars length: {len(bars)}")
            except Exception as e:
                print(f"Could not retrieve data for {ticker}..", end = " | ")
                print(e)
                print("Trying to reconnect to server..")
                
            else: 
                df = pd.DataFrame(bars)
                return df 
            
            finally: 
                # wait 5 seconds before trying to reconnect
                time.sleep(10)
                attempt += 1
                print(f"Attempt: {attempt}")
                # Try to reconnect (assuming it was the connection error).
                self.connect()
                
                # stop the programm if max attempts reached
                if attempt >= max_attempts: 
                    print(f"Programm stopped: max attempts reached..")
                    exit(0)
    def get_path(self, ticker):
        symbol_folder = os.path.join(self.hist_data_folder, ticker)
        folder_contents = os.listdir(symbol_folder)
        if f"{ticker}.parquet.gzip" in folder_contents:
            data_path = os.path.join(symbol_folder, f"{ticker}.parquet.gzip")
            return data_path
        elif f"{ticker}.csv" in folder_contents:
            data_path = os.path.join(symbol_folder, f"{ticker}.csv")
            return data_path
        else:
            print(f"ERROR: Could not find any data for {ticker}..")
            exit(0)

    def load_data(self, ticker):
        data_path = self.get_path(ticker)
        _, file_extension = os.path.splitext(data_path)
        if file_extension == ".gzip":
            return pd.read_parquet(data_path)
        else:
            return pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")

    def update_hist_data(self, tickers):
        for ticker in tickers:
            df = self.load_data(ticker)
            last_timestamp = df.index[-1]

    def retrieve_all_historical_data(self):  
        total_start_t = time.time()
        for ticker in tqdm.tqdm(self.tickers): 
            try: 
                start_t = time.time()
                df = self.get_ticker_historical_data(ticker)

                path = os.path.join(self.folder, ticker)
                self.create_ticker_folder(ticker)
                start = pd.to_datetime(df.iloc[0, 0], unit = "ms") 
                end = pd.to_datetime(df.iloc[-1, 0], unit = "ms")
                self.store_df(df, path)
                print(f"Retrieved in total {len(df)} samples from {start} to {end}..")
                #self.remove_ticker_and_store_list(ticker)
                end_t = time.time()
                print(f"Retrieving data for {ticker} took {(end_t - start_t)/60} mins..")
                print("="*80)
                time.sleep(1)
            except Exception as e:
                print(f"Error {e} for ticker {ticker}..")
                continue
                
        total_end_t = time.time()
        print(f"Retrieving all ticker historical data took in total {(total_end_t - total_start_t)/60} mins..")
        
        
if __name__ == "__main__":
    symbols = ["ADAUSDT", "SOLUSDT", "DOTUSDT", "XRPUSDT"]
    dataRetriever = DataRetriever()
    # connect client to binance server
    dataRetriever.connect()
    # get all available coins traidable with USDT (default)
    dataRetriever.tickers = symbols
    #dataRetriever.get_only_tickers_containing_symbol()
    # get all historical data for available coins 
    dataRetriever.retrieve_all_historical_data()
