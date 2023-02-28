from binance.client import Client
import os
import pandas as pd
import numpy as np
import pickle
#import matplotlib.pyplot as plt
from binance.client import Client
from binance import ThreadedWebsocketManager
import pandas as pd
from datetime import datetime, timedelta

#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import time
#Disable the warnings
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../utilities')
sys.path.append('../ml_backtesting')
from Email import Email

from MlDataManager import MlDataManager
from datetime import datetime



from credentials import *
testnet = True




class LongOnlyTrader():

    def __init__(self, symbol, bar_length, units, position=0, testnet=True):

        self.symbol = symbol
        self.bar_length = bar_length
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d",
                                    "1w", "1M"]
        self.units = units
        self.position = position
        self.trades = 0
        self.trade_values = []

        self.model_path = '../utilities/xrp_model.sav'
        # *****************add strategy-specific attributes here******************

        self.data_manager = MlDataManager(training=False)
        self.load_model()

        self.email = Email()
        # ************************************************************************
        # parameter used for binance client connection/reconnection
        self.client = None
        self.testnet = testnet
        self.max_attempts = 100
        self.wait = 15
        self.wait_increase = 15

        # create and connect binance client
        self.connect_client()

    def connect_client(self):
        try:
            if not self.testnet:
                self.client = Client(api_key=api_key, api_secret=secret_key, tld="com")
            else:
                self.client = Client(api_key=testnet_api_key, api_secret=testnet_secret_key, tld="com", testnet=True)
        except Exception as e:
            print(f"Error occurred during connection establishment with server..")

    def client_is_connected(self, reconnect=True, verbose=False):

        try:
            if len(self.client.ping()) == 0:
                if verbose:
                    print("Connection is ok.")
                return True
        except Exception as e:
            if verbose:
                print(f"Connection with Binance server was lost.. Trying to reconnect..")
            if reconnect:
                return self.reconnect_client()
            else:
                return False

    def reconnect_client(self):
        attempt = 0
        success = False
        self.max_attempts = 35
        self.wait = 15
        self.wait_increase = 15
        while True:
            try:
                self.connect_client()
                success = self.client_is_connected(reconnect=False, verbose=True)
            except Exception as e:
                print(e, end=" | ")
            else:
                if success:
                    print("INFO: Reconnected. Starting trading..")
                    self.start_trading()
                    break
            finally:
                attempt += 1
                print("Attempt: {}".format(attempt), end='\n')
                if not success:
                    if attempt >= self.max_attempts:
                        print("Max_attempts reached!")
                        print("Could not connect with Binance server.")
                        exit(0)
                    else:  # try again
                        time.sleep(self.wait)
                        self.wait += self.wait_increase
                        self.wait_increase += 15

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = pickle.load(open(self.model_path, 'rb'))
        else:
            print(f"ERROR: The model path {self.model_path} does not exist..")

    def start_trading(self, historical_days=3):

        self.twm = ThreadedWebsocketManager()
        self.twm.start()

        if self.bar_length in self.available_intervals:
            self.get_most_recent(symbol=self.symbol, interval=self.bar_length,
                                 days=historical_days)
            self.twm.start_kline_socket(callback=self.stream_candles,
                                        symbol=self.symbol, interval=self.bar_length)
        # "else" to be added later in the course

    def get_most_recent(self, symbol, interval, days):

        now = datetime.utcnow()
        past = str(now - timedelta(days=days))

        bars = self.client.get_historical_klines(symbol=symbol, interval=interval,
                                            start_str=past, end_str=None, limit=1000)
        df = pd.DataFrame(bars)
        df["Date"] = pd.to_datetime(df.iloc[:, 0], unit="ms")
        df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                      "Clos Time", "Quote Asset Volume", "Number of Trades",
                      "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.set_index("Date", inplace=True)
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["Complete"] = [True for row in range(len(df) - 1)] + [False]

        self.data = df

    def stream_candles(self, msg):

        # extract the required items from msg
        event_time = pd.to_datetime(msg["E"], unit="ms")
        start_time = pd.to_datetime(msg["k"]["t"], unit="ms")
        first = float(msg["k"]["o"])
        high = float(msg["k"]["h"])
        low = float(msg["k"]["l"])
        close = float(msg["k"]["c"])
        volume = float(msg["k"]["v"])
        complete = msg["k"]["x"]

        # stop trading session
        # if event_time >= datetime(2021, 11, 4, 9, 55):
        #   self.twm.stop()
        #  if self.position != 0:
        #     order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
        #    self.report_trade(order, "GOING NEUTRAL AND STOP")
        #   self.position = 0
        # else:
        #   print("STOP")

        # more stop examples:
        # if self.trades >= xyz
        # if self.cum_profits <> xyz

        # print out
        #print(".", end="", flush=True)  # just print something to get a feedback (everything OK)

        # feed df (add new bar / update latest bar)
        self.data.loc[start_time] = [first, high, low, close, volume, complete]

        # prepare features and define strategy/trading positions whenever the latest bar is complete
        if complete == True:
            if self.ml_trading:
                self.data_manager.set_data(self.data)
                self.data_manager.preprocess_data()
                self.define_strategy()
                self.execute_trades()
            else:
                pass
            try:
                self.report_as_email()
            except Exception as e:
                print(f"ERROR: {e}..")
                print(f"Could not send email report..")

    def report_as_email(self):
        now = datetime.now()
        # If it's the beginning of the hour, send the email
        if now.minute == 0:
            if self.trades % 2 == 0:
                real_profit = round(np.sum(self.trade_values[-2:]), 3)
                self.cum_profits = round(np.sum(self.trade_values), 3)
            else:
                real_profit = 0
                self.cum_profits = round(np.sum(self.trade_values[:-1]), 3)
            text = f"""Number of Trades:  {self.trades}
    Profit = {real_profit} | CumProfits = {self.cum_profits}"""
            self.email.send_email(text)

    def define_strategy(self):

        # standardization
        # predict
        self.prepared_data = self.data_manager.data.copy()
        self.prepared_data.drop(columns=["Complete"], inplace=True)
        self.prepared_data["position"] = self.model.predict(self.prepared_data)
        self.prepared_data["position"] = self.prepared_data.position.ffill().fillna(0)  # start with neutral position if no strong signal

    def get_trading_signal(self) -> str:
        signal = self.model.get_signal(self.data)
        return signal

    def execute_trades(self):
        if self.prepared_data["position"].iloc[-1] == 1:  # if position is long -> go/stay long
            if self.position == 0:
                order = self.client.create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.prepared_data["position"].iloc[-1] == 0:  # if position is neutral -> go/stay neutral
            if self.position == 1:
                order = self.client.create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING NEUTRAL")
            self.position = 0

    def report_trade(self, order, going):

        # extract data from order object
        side = order["side"]
        time = pd.to_datetime(order["transactTime"], unit="ms")
        base_units = float(order["executedQty"])
        quote_units = float(order["cummulativeQuoteQty"])
        price = round(quote_units / base_units, 5)

        # calculate trading profits
        self.trades += 1
        if side == "BUY":
            self.trade_values.append(-quote_units)
        elif side == "SELL":
            self.trade_values.append(quote_units)

        if self.trades % 2 == 0:
            real_profit = round(np.sum(self.trade_values[-2:]), 3)
            self.cum_profits = round(np.sum(self.trade_values), 3)
        else:
            real_profit = 0
            self.cum_profits = round(np.sum(self.trade_values[:-1]), 3)

        # print trade report
        print(2 * "\n" + 100 * "-")
        print("{} | {}".format(time, going))
        print("{} | Base_Units = {} | Quote_Units = {} | Price = {} ".format(time, base_units, quote_units, price))
        print("{} | Profit = {} | CumProfits = {} ".format(time, real_profit, self.cum_profits))
        print(100 * "-" + "\n")


def record_order(order, trader):
    side = order["side"]
    quote_units = float(order["cummulativeQuoteQty"])
    # calculate trading profits
    trader.trades += 1
    if side == "BUY":
        trader.trade_values.append(-quote_units)
    elif side == "SELL":
        trader.trade_values.append(quote_units)

if __name__ == "__main__":
    symbol = "XRPUSDT"
    bar_length = "15m"
    units = 1000
    position = 0

    trader = LongOnlyTrader(symbol=symbol, bar_length=bar_length, units=units, position=position)
    trader.start_trading(historical_days=4)

    print("Trading started..")
    while True:
        trader.client_is_connected(reconnect=True, verbose=False)
        time.sleep(15)