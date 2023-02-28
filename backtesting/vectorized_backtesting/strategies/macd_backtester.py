import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
import os
import sys
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utilities.logger import logger

from utilities.data_plot import DataPlot
from utilities.performance import Performance
from itertools import product

from backtester_base import VectorBacktesterBase

plt.style.use("seaborn")


class MACDBacktester(VectorBacktesterBase):
    ''' Class for the vectorized backtesting of MACD-based trading strategies.

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    EMA_S: int
        time window in days for shorter EMA
    EMA_L: int
        time window in days for longer EMA
    signal_mw: int
        time window is days for MACD Signal
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    tc: float
        proportional transaction costs per trade


    Methods
    =======
    get_data:
        retrieves and prepares the data

    set_parameters:
        sets new MACD parameter(s)

    test_strategy:
        runs the backtest for the MACD-based strategy

    plot_results:
        plots the performance of the strategy compared to buy and hold

    update_and_run:
        updates MACD parameters and returns the negative absolute performance (for minimization algorithm)

    optimize_parameters:
        implements a brute force optimization for the three MACD parameters
    '''

    def __init__(self, filepath, symbol, tc=0.00007, dataset="training", start=None, end=None):
        super().__init__(filepath=filepath, symbol=symbol, start=start, end=end, tc=tc, dataset=dataset)
        self.indicator = "MACD"
        self.perf_obj = Performance(symbol=symbol)
        self.dataploter = DataPlot(dataset, self.indicator, self.symbol)

    def __repr__(self):
        return "MACDBacktester(symbol = {}, MACD({}, {}, {}), start = {}, end = {})".format(self.symbol, self.EMA_S,
                                                                                            self.EMA_L, self.signal_mw,
                                                                                            self.start, self.end)


    def test_strategy(self, freq=5, EMA_S=50, EMA_L=200, signal_mw=9):  # Adj!!!
        '''
        Prepares the data and backtests the trading strategy incl. reporting (Wrapper).

        Parameters
        ============
        freq: int
            data frequency/granularity to work with (in minutes)

        window: int
            time window (number of bars) to calculate the simple moving average price (SMA).

        dev: int
            number of standard deviations to calculate upper and lower bands.
        '''
        self.freq = "{}min".format(freq)
        self.EMAS = EMA_S
        self.SMA_L = EMA_L  # NEW!!!

        self.prepare_data(freq, EMA_S, EMA_L, signal_mw)
        self.upsample()
        self.run_backtest()

        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data


        # store the test results in the dataframe
        # set strategy data and calculate performance
        self.perf_obj.set_data(self.results)
        self.perf_obj.calculate_performance()
        # store strategy performance data for further plotting
        params_dic = {"freq": freq, "EMA_S": EMA_S, "EMA_L": EMA_L, "signal_mw": signal_mw}
        self.dataploter.store_testcase_data(self.perf_obj, params_dic)  # comb[0] is current data freq
        #self.print_performance()


    def prepare_data(self, freq, ema_s_val, ema_l_val, signal_mw):
        ''' Prepares the Data for Backtesting.
           '''
        freq = f"{freq}min"
        data_resampled = self.data.resample(freq).last().dropna().iloc[:-1].copy()
        data_resampled["returns"] = np.log(1 + data_resampled.Close.pct_change())
        data_resampled.dropna()
        ######### INSERT THE STRATEGY SPECIFIC CODE HERE ##################

        ema_s = data_resampled["Close"].ewm(span=ema_s_val-1, min_periods=ema_s_val).mean()
        ema_l = data_resampled["Close"].ewm(span=ema_l_val-1, min_periods=ema_l_val).mean()
        macd = (ema_s - ema_l)
        macd_signal = macd.ewm(span=signal_mw, min_periods=signal_mw).mean()
        position = np.where(macd > macd_signal, 1, 0) \
            #(macd > macd_signal).astype(int).ffill().fillna(0)
        ###################################################################
        data_resampled = data_resampled.assign(position=position).dropna()
        self.results = data_resampled


    def prepare_data(self, freq, ema_s_val, ema_l_val, signal_mw):  # Adj!!!
        ''' Prepares the Data for Backtesting.
        '''
        freq = f"{freq}min"
        data_resampled = self.data.resample(freq).last().dropna().iloc[:-1].copy()
        data_resampled["returns"] = np.log(1 + data_resampled.Close.pct_change())
        data_resampled.dropna()
        ######### INSERT THE STRATEGY SPECIFIC CODE HERE ##################

        ema_s = data_resampled["Close"].ewm(span=ema_s_val - 1, min_periods=ema_s_val).mean()
        ema_l = data_resampled["Close"].ewm(span=ema_l_val - 1, min_periods=ema_l_val).mean()
        macd = (ema_s - ema_l)
        macd_signal = macd.ewm(span=signal_mw, min_periods=signal_mw).mean()
        position = np.where(macd > macd_signal, 1, 0)#.astype(int).ffill().fillna(0)
        ###################################################################
        data_resampled = data_resampled.assign(position=position).dropna()
        self.results = data_resampled

    def optimize_strategy(self, freq_range, ema_s_range, ema_l_range, signal_mw_range, metric="Multiple"):  # Adj!!!
        '''
        Backtests strategy for different parameter values incl. Optimization and Reporting (Wrapper).

        Parameters
        ============
        freq_range: tuple
            tuples of the form (start, end, step size).

        window_range: tuple
            tuples of the form (start, end, step size).

        dev_range: tuple
            tuples of the form (start, end, step size).

        metric: str
            performance metric to be optimized (can be: "Multiple", "Sharpe", "Sortino", "Calmar", "Kelly")
        '''

        self.data = self.data.loc[:, ["Close"]]
        # performance_function = self.perf_obj.performance_functions[metric]

        freqs = range(*freq_range)
        ema_s_range = range(*ema_s_range)
        ema_l_range = range(*ema_l_range)  # NEW!!!
        signal_mw_range = range(*signal_mw_range)
        ema_pairs = [(ema_s, ema_l) for ema_s in ema_s_range for ema_l in ema_l_range if ema_l > ema_s and (ema_l - ema_s) >= 10]

        combinations = list(product(freqs, ema_pairs, signal_mw_range))
        #combinations = list(product(freqs, ema_s, ema_l, signal_mw))
        #combinations = [(_, ema_s, ema_l, _) for (_, ema_s, ema_l, _) in combinations if ema_l > ema_s]
        logger.info(f"macd_backtester: Optimizing of {self.indicator} for {self.symbol} using in total {len(combinations)} combinations..")
        #performance = []

        # for data plotting
        for (freq, (ema_s, ema_l), signal_mw) in tqdm(combinations):
            self.prepare_data(freq, ema_s, ema_l, signal_mw)
            if metric != "Multiple":
                self.upsample()
            self.run_backtest()
            #performance.append(performance_function(self.results.strategy))
            # set strategy data and calculate performance
            self.perf_obj.set_data(self.results)
            self.perf_obj.calculate_performance()
            # store strategy performance data for further plotting
            params_dic = {"freq": freq, "SMA_S": ema_s, "SMA_L": ema_l, "Signal_Mw": signal_mw}
            self.dataploter.store_testcase_data(self.perf_obj, params_dic)

    def find_best_strategy(self):
        ''' Finds the optimal strategy (global maximum) given the parameter ranges.
        '''
        best = self.results_overview.nlargest(1, "Performance")
        freq = int(best.Freq.iloc[0])
        sma_s = int(best.SMA_S.iloc[0])
        sma_l = best.SMA_L.iloc[0]  # NEW!!!
        perf = best.Performance.iloc[0]
        print("Frequency: {} | EMA_S: {} | EMA_L: {} | {}: {}".format(freq, sma_s, sma_l, self.metric,
                                                                      round(perf, 6)))
        self.test_strategy(freq, sma_s, sma_l)