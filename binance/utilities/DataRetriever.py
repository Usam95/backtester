import os 
import pandas as pd
import numpy as np
import tqdm
import pickle
import sys
import time

sys.path.append(".")

from credentials import *
from binance.client import Client


class DataRetriever: 
    
    def __init__(self, folder="../hist_data"):
        self.folder = "../hist_data"
        print(self.folder)
    def connect(self): 
        self.client = Client(api_key = api_key, api_secret = secret_key, tld = "com")
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

        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors = "coerce")

        symbol = path.split("/")[-1]
        path = path + "/" + symbol + ".csv"
        print(f"Storing data for {symbol} as {path}")
        df.to_csv(path)                                       
        
    def get_ticker_historical_data(self, ticker):
        print(f"Retrieving data for {ticker} with 1m interval..")
        timestamp = self.client._get_earliest_valid_timestamp(symbol = ticker, interval = "1m")
        start_t = pd.to_datetime(timestamp, unit = "ms") # earliest data available on Binance
        print(f"Data for {ticker} are available from {start_t}..")
        #timestamp = "2022-01-03"
        bars = self.client.get_historical_klines(symbol = ticker, interval = "1m", start_str = timestamp)
        df = pd.DataFrame(bars)
        return df                                   
                                           
    def retrieve_all_historical_data(self):  
        total_start_t = time.time()
        for ticker in tqdm.tqdm(self.tickers): 
            try: 
                start_t = time.time()
                df = self.get_ticker_historical_data(ticker)

                path = os.path.join(self.folder, ticker)
                self.create_ticker_folder(ticker)
                self.store_df(df, path)
                self.remove_ticker_and_store_list(ticker)
                end_t = time.time()
                print(f"Retrieving data for {ticker} took {(end_t - start_t)/60} mins..")
                print("="*80)
                time.sleep(1)
            except Exception as e:
                print(f"Error {e} for ticker {ticker}..")
                continue
                
        total_end_t = time.time()
        print(f"Retrieving all ticker historical data took in total {(total_end_t - total_start_t)/60} mins..")