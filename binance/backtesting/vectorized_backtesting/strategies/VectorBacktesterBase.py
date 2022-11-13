import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split


class VectorBacktesterBase:

    def __init__(self, filepath, symbol, tc=0.00007, start=None, end=None, dataset="training"):
        self.filepath = filepath
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.results = None
        self.dataset = dataset
        self.test_size = 0.2
        self.get_data()
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))
        self.results_folder = "../results"
    def __repr__(self):
        return "BollBacktester(symbol = {}, start = {}, end = {})".format(self.symbol, self.start, self.end)

    def get_data(self):
        ''' Imports the data.
        '''

        _, file_extension = os.path.splitext(self.filepath)
        if file_extension == ".gzip":
            raw = pd.read_parquet(self.filepath)
        else:
            raw = pd.read_csv(self.filepath, parse_dates=["Date"], index_col="Date")

        raw["returns"] = np.log(raw.Close / raw.Close.shift(1))
        if 'Unnamed: 0' in raw.columns:
            raw.drop(columns=['Unnamed: 0'], inplace=True)
        self.data = raw

        if self.start is not None and self.end is not None:
            self.data = self.data.loc[self.start:self.end].copy()
        else:
            self.train_test_split()

    def train_test_split(self):
        print(f"Length bevore splitting: {len(self.data)}")
        if self.dataset == "training":
            self.data, test = train_test_split(self.data, test_size=0.2, shuffle = False)
            print(f"Using training set: Length after splitting: {len(self.data)}")
            print(f"Training Dataset from {self.data.index[0]} to {self.data.index[-1]}")
            print(f"="*90)
            print(f"Testing set: Length after splitting: {len(test)}")
            print(f"Testing Dataset from {test.index[0]} to {test.index[-1]}")

        elif self.dataset == "testing":
            _, self.data = train_test_split(self.data, test_size=0.2, shuffle = False)
            print(f"Using testing set: Length after splitting: {len(self.data)}")
            print(f"Dataset from {self.data.index[0]} to {self.data.index[-1]}")
        else:
            print(f"ERROR: Please specify the dataset to be used.")

    def select_data(self, start, end):
        ''' Selects sub-sets of the financial data. '''
        self.data = self.data[(self.data.index >= start) & (self.data.index <= end)].copy()

    def run_backtest(self):
        ''' Runs the strategy backtest.
        '''
        data = self.results.copy()
        data["strategy"] = data["position"].shift(1) * data["returns"]

        # determine the number of trades in each bar
        data["trades"] = data.position.diff().fillna(0).abs()

        # subtract transaction/trading costs from pre-cost return
        data.strategy = data.strategy - data.trades * self.tc

        self.results = data

    def upsample(self):
        '''  Upsamples/copies trading positions back to higher frequency.
        '''

        data = self.data.copy()
        resamp = self.results.copy()

        data["position"] = resamp.position.shift()
        data = data.loc[resamp.index[0]:].copy()
        data.position = data.position.shift(-1).ffill()
        data.dropna(inplace=True)
        self.results = data

    def plot_results(self, leverage=False):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        elif leverage:
            title = "{} | Window = {} | Frequency = {} | TC = {} | Leverage = {}".format(self.symbol, self.window,
                                                                                         self.freq, self.tc,
                                                                                         self.leverage)
            self.results[["creturns", "cstrategy", "cstrategy_levered"]].plot(title=title, figsize=(12, 8))
        else:
            title = "{} | Window = {} | Frequency = {} | TC = {}".format(self.symbol, self.window, self.freq, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))

    def visualize_many(self):
        ''' Plots parameter values vs. Performance.
        '''
        if self.results_overview is None:
            print("Run optimize_strategy() first.")
        else:
            matrix = self.results_overview.reset_index().pivot(index="Freq", columns="Windows", values="Performance")

            plt.figure(figsize=(12, 8))
            sns.set_theme(font_scale=1.5)
            sns.heatmap(matrix, cmap="RdYlGn", robust=True, cbar_kws={"label": "{}".format(self.metric)})
            plt.show()

    def add_sessions(self, visualize=False):
        '''
        Adds/Labels Trading Sessions and their compound returns.

        Parameter
        ============
        visualize: bool, default False
            if True, visualize compound session returns over time
        '''

        if self.results is None:
            print("Run test_strategy() first.")

        data = self.results.copy()
        data["session"] = np.sign(data.trades).cumsum().shift().fillna(0)
        data["session_compound"] = data.groupby("session").strategy.cumsum().apply(np.exp) - 1
        self.results = data
        if visualize:
            data["session_compound"].plot(figsize=(12, 8))
            plt.show()

    def add_stop_loss(self, sl_thresh, report=True):
        '''
        Adds Stop Loss to the Strategy.

        Parameter
        ============
        sl_thresh: float (negative)
            maximum loss level in % (e.g. -0.02 for -2%)

        report: bool, default True
            if True, print Performance Report incl. Stop Loss.
        '''

        self.sl_thresh = sl_thresh

        if self.results is None:
            print("Run test_strategy() first.")

        self.add_sessions()
        self.results = self.results.groupby("session").apply(self.define_sl_pos)
        self.run_backtest()
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        self.add_sessions()

        if report:
            self.print_performance()

    def add_take_profit(self, tp_thresh, report=True):
        '''
        Adds Take Profit to the Strategy.

        Parameter
        ============
        tp_thresh: float (positive)
            maximum profit level in % (e.g. 0.02 for 2%)

        report: bool, default True
            if True, print Performance Report incl. Take Profit.
        '''
        self.tp_thresh = tp_thresh

        if self.results is None:
            print("Run test_strategy() first.")

        self.add_sessions()
        self.results = self.results.groupby("session").apply(self.define_tp_pos)
        self.run_backtest()
        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        self.add_sessions()

        if report:
            self.print_performance()

    def define_sl_pos(self, group):
        if (group.session_compound <= self.sl_thresh).any():
            start = group[group.session_compound <= self.sl_thresh].index[0]
            stop = group.index[-2]
            group.loc[start:stop, "position"] = 0
            return group
        else:
            return group

    def define_tp_pos(self, group):
        if (group.session_compound >= self.tp_thresh).any():
            start = group[group.session_compound >= self.tp_thresh].index[0]
            stop = group.index[-2]
            group.loc[start:stop, "position"] = 0
            return group
        else:
            return group

    def print_performance(self, perf_obj, **kwargs):
        ''' Calculates and prints various Performance Metrics.
        '''

        data = self.results.copy()

        to_analyze = data.strategy

        strategy_multiple = round(perf_obj.calculate_multiple(to_analyze), 6)
        bh_multiple = round(perf_obj.calculate_multiple(data.returns), 6)
        outperf = round(strategy_multiple - bh_multiple, 6)
        cagr = round(perf_obj.calculate_cagr(to_analyze), 6)
        ann_mean = round(perf_obj.calculate_annualized_mean(to_analyze), 6)
        ann_std = round(perf_obj.calculate_annualized_std(to_analyze), 6)
        sharpe = round(perf_obj.calculate_sharpe(to_analyze), 6)
        sortino = round(perf_obj.calculate_sortino(to_analyze), 6)
        max_drawdown = round(perf_obj.calculate_max_drawdown(to_analyze), 6)
        calmar = round(perf_obj.calculate_calmar(to_analyze), 6)
        max_dd_duration = round(perf_obj.calculate_max_dd_duration(to_analyze), 6)
        kelly_criterion = round(perf_obj.calculate_kelly_criterion(to_analyze), 6)

        print(100 * "=")
        print("SIMPLE CONTRARIAN STRATEGY | INSTRUMENT = {} | Freq= {} | EMA_S = {} | EMA_L".format(kwargs["ticker"],
                                                                                             kwargs["freq"],
                                                                                             kwargs["ema_s"],
                                                                                             kwargs["ema_l"]))
        print(100 * "-")
        # print("\n")
        print("PERFORMANCE MEASURES:")
        print("\n")
        print("Multiple (Strategy):         {}".format(strategy_multiple))
        print("Multiple (Buy-and-Hold):     {}".format(bh_multiple))
        print(38 * "-")
        print("Out-/Underperformance:       {}".format(outperf))
        print("\n")
        print("CAGR:                        {}".format(cagr))
        print("Annualized Mean:             {}".format(ann_mean))
        print("Annualized Std:              {}".format(ann_std))
        print("Sharpe Ratio:                {}".format(sharpe))
        print("Sortino Ratio:               {}".format(sortino))
        print("Maximum Drawdown:            {}".format(max_drawdown))
        print("Calmar Ratio:                {}".format(calmar))
        print("Max Drawdown Duration:       {} Days".format(max_dd_duration))
        print("Kelly Criterion:             {}".format(kelly_criterion))

        print(100 * "=")