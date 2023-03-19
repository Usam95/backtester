'''
Created on 07.11.2021

@author: usam
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv


class IterativeBase:

    def __init__(self, config):
        pass
    def set_interval(self, interval):
        self.interval = interval

    def set_units(self, units):
        self.units = units

    def reset(self, units):
        self.units = units
        self.get_data()
        self.init_initial_balance()
        self.trades = 0
        self.position = 0
        self.position_info = {}
        self.take_profit_pct = 0.1
        self.stop_loss_pct = 0.05

        self.buy_price = 0
        self.sell_price = 0
        self.perf = 0
        # self.current_balance = 0

        self.trade_values = []

    def set_symbol(self, symbol):
        self.symbol = symbol

    def set_path(self, path):
        self.path = path
        # self.get_data()

    def set_strategy(self, strategy_name):
        self.strategy = strategy_name

    def create_results_file(self):

        with open('./backtest_results.csv', 'w') as f:
            writer = csv.writer(f)
            header = ['Symbol', 'Strategy', 'Initial Balance', 'Final Balance', 'Performance', 'Trades']
            writer.writerow(header)

    def update_current_df(self):

        if len(self.df) > (self.end_idx + 1):
            self.cur_df = self.df.iloc[self.start_idx: self.end_idx]
            self.start_idx += 1
            self.end_idx += 1
        else:
            print("Could not select a subset from the dataframe..")

    def init_initial_balance(self, price=None):

        self.first_price = self.df["Close"].iloc[(self.end_idx - 1)]
        if price:
            self.initial_balance = price
            self.current_balance = price
        else:
            self.initial_balance = round(self.first_price * self.units, 2) + (self.units * self.first_price * 0.2)
            self.current_balance = self.initial_balance
        # self.initial_balance += 200

    def get_data(self):
        # self.df = pd.read_csv(self.path, parse_dates=["Date"]).dropna()

        self.df = pd.read_csv(self.path)
        # self.df = self.df[:12000]
        optim_data_len = int(len(self.df) * 0.4)

        self.strategy_df = self.df[:optim_data_len]
        self.df = self.df[optim_data_len:]

        self.start_idx = 0
        self.end_idx = 201
        # self.df = self.df[-500:]
        # raw.rename(columns={ "askopen": "Open", "askclose": "Close", "askhigh": "High", "asklow": "Low"}, errors="raise", inplace=True)
        # self.df.set_index("Date",inplace=True)
        # raw["price"] = raw["Close"]
        # raw = raw.loc[self.start:self.end]
        # raw["returns"] = np.log(raw.price / raw.price.shift(1))
        # self.df = raw

    def plot_data(self, cols=None):
        if cols is None:
            cols = "price"
        self.data[cols].plot(figsize=(12, 8), title=self.symbol)

    def get_values(self, bar):
        date = str(self.data.index[bar].date())
        price = round(self.data.price.iloc[bar], 5)
        # volume = round(self.data.Volume.iloc[bar], 5)
        return date, price  # ,volume

    def print_current_balance(self, bar):
        date, price = self.get_values(bar)
        print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))

    def buy_instrument(self, price):

        # Calculate fees first:
        fees = (price / 100) * self.tc
        # date, price = self.get_values(bar)
        date = str(self.df["Date"].iloc[self.end_idx])
        self.buy_price = price
        # units = int(self.current_balance / price)
        # =======================================================================
        # if self.current_balance < price:
        #     print(f"ERROR: Current balance {round(self.current_balance, 2)} is not sufficient to buy for price: {price}.")
        #     self.price_error += 1
        # =======================================================================
        self.current_balance -= self.units * price  # reduce cash balance by "purchase price"
        self.current_balance -= fees
        self.trade_values.append(-(self.units * price))
        # self.units += units
        self.trades += 1
        # self.current_balance -= 1
        # print("{} |  Buying {} for {}".format(date, self.units, round(price*self.units, 5)))

    def sell_instrument(self, price):
        # Calculate fees first:
        fees = (price / 100) * self.tc

        date = str(self.df["Date"].iloc[self.end_idx])
        # print("{} |  Selling {} for {}".format(date, self.units, round(price*self.units, 5)))

        self.current_balance += self.units * price  # increases cash balance by "purchase price"
        self.current_balance -= fees

        self.trade_values.append((self.units * price))
        # self.units = 0
        self.trades += 1
        self.sell_price = price
        # self.current_balance -= 1

    def print_current_position_value(self, bar):
        date, price = self.get_values(bar)
        cpv = self.units * price
        print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))

    def print_current_nav(self, bar):
        date, price = self.get_values(bar)
        nav = self.current_balance + self.units * price
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))

    def close_pos(self, price):
        self.last_price = price
        # date, price = self.get_values(bar)
        date = str(self.df["Date"].iloc[self.end_idx])
        print(75 * "-")
        # print("{} | +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * price  # closing final position (works with short and long!)
        print("{} || closing position of {} for {}".format(date, self.units, price))
        # self.units = 0 # setting position to neutral
        self.trades += 1
        self.perf = round((self.current_balance - self.initial_balance) / self.initial_balance * 100, 2)

        self.finish_capital = round(self.last_price * self.units, 2) + (self.units * self.first_price * 0.2)
        self.buy_hold = round((self.finish_capital - self.initial_balance) / self.initial_balance * 100, 2)

        self.trade_values.append(price)
        print(f"{date} || Current Balance: {self.current_balance}")
        print(f"Initial Balance: {self.initial_balance}")
        print("net performance (%) = {}".format(self.perf, 2))
        print("number of trades executed = {}".format(self.trades))
        print(f"Number of price errors: {self.price_error}")
        print(75 * "-")

    def store_results(self):
        with open('./backtest_results.csv', 'a') as f:
            writer = csv.writer(f)
            results = []
            results.append(self.symbol)
            results.append(self.strategy)
            results.append(self.interval)
            results.append(self.first_price)
            results.append(self.last_price)
            results.append(round(self.initial_balance, 2))
            results.append(round(self.current_balance, 2))
            results.append(self.buy_hold)
            results.append(self.perf)
            results.append(round(self.perf - self.buy_hold, 2))
            results.append(self.trades)

            writer.writerow(results)


class VectorBacktesterBase:
    pass