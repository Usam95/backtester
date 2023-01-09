from binance.client import Client
import os
import pandas as pd
import numpy as np
import tqdm
import pickle
import matplotlib.pyplot as plt
from binance.client import Client
from binance import ThreadedWebsocketManager
import pandas as pd
from datetime import datetime, timedelta

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
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

if not testnet:
    client = Client(api_key = api_key, api_secret = secret_key, tld = "com")
else:
    client = Client(api_key = testnet_api_key, api_secret = testnet_secret_key, tld = "com", testnet=True)


class LongOnlyTrader():

    def __init__(self, symbol, bar_length, units, position=0):

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

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = pickle.load(open(self.model_path, 'rb'))
        else:
            print(f"ERROR: The model path {self.model_path} does not exist..")

    def start_trading(self, historical_days):

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

        bars = client.get_historical_klines(symbol=symbol, interval=interval,
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
        print(".", end="", flush=True)  # just print something to get a feedback (everything OK)

        # feed df (add new bar / update latest bar)
        self.data.loc[start_time] = [first, high, low, close, volume, complete]

        # prepare features and define strategy/trading positions whenever the latest bar is complete
        if complete == True:
            self.data_manager.set_data(self.data)
            self.data_manager.preprocess_data()
            self.define_strategy()
            self.execute_trades()
            self.report_as_email()

    def report_as_email(self):
        now = datetime.now()
        # If it's the beginning of the hour, send the email
        #if now.minute == 0:
        cum_profits = round(np.sum(self.trade_values[:-1]), 3)
        num_trades = self.trades

        text = f"""Total number of trades: {num_trades}
Total profit: {cum_profits}
        """
        self.email.send_email(text)

    def define_strategy(self):

        # standardization
        # predict
        self.prepared_data = self.data_manager.data.copy()
        self.prepared_data.drop(columns=["Complete"], inplace=True)
        self.prepared_data["position"] = self.model.predict(self.prepared_data)
        self.prepared_data["position"] = self.prepared_data.position.ffill().fillna(
            0)  # start with neutral position if no strong signal

    def execute_trades(self):
        if self.prepared_data["position"].iloc[-1] == 1:  # if position is long -> go/stay long
            if self.position == 0:
                order = client.create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.prepared_data["position"].iloc[-1] == 0:  # if position is neutral -> go/stay neutral
            if self.position == 1:
                order = client.create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
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

if __name__ == "__main__":
    symbol = "XRPUSDT"
    bar_length = "1m"
    units = 500
    position = 0

    trader = LongOnlyTrader(symbol=symbol, bar_length=bar_length, units=units, position=position)
    trader.start_trading(historical_days=1)
    print("Waiting 60 sec. before finishing..")
    time.sleep(500)