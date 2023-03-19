import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from utilities.data_plot import DataPlot
from utilities.performance import Performance
from backtester_base import VectorBacktesterBase
from utilities.logger import Logger

logger = Logger().get_logger()


class SMABacktester(VectorBacktesterBase):

    def __init__(self, filepath, symbol, tc=0.00007, dataset="training", start=None, end=None):
        super().__init__(filepath=filepath, symbol=symbol, start=start, end=end, tc=tc, dataset=dataset)
        self.indicator = "SMA"
        self.perf_obj = Performance(symbol=symbol)
        self.dataploter = DataPlot(dataset, self.indicator, self.symbol)

    def test_strategy(self, freq=5, SMA_S=50, SMA_L=200):  # Adj!!!
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
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L  # NEW!!!

        self.prepare_data(freq, SMA_S, SMA_L)
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
        params_dic = {"freq": freq, "SMA_S": SMA_S, "SMA_L": SMA_L}
        self.dataploter.store_testcase_data(self.perf_obj, params_dic)  # comb[0] is current data freq
        #self.print_performance()

    def prepare_data(self, freq, sma_s_val, sma_l_val):  # Adj!!!
        ''' Prepares the Data for Backtesting.
        '''
        freq = f"{freq}min"
        data_resampled = self.data.resample(freq).last().dropna().iloc[:-1].copy()
        data_resampled["returns"] = np.log(1 + data_resampled.Close.pct_change())
        data_resampled.dropna()

        ######### INSERT THE STRATEGY SPECIFIC CODE HERE ##################
        ema_s = data_resampled["Close"].rolling(window=sma_s_val, min_periods=sma_s_val).mean()
        ema_l = data_resampled["Close"].rolling(window=sma_l_val, min_periods=sma_l_val).mean()
        position = np.zeros(len(data_resampled))
        position[ema_s > ema_l] = 1
        ###################################################################
        position = pd.Series(position, index=data_resampled.index).ffill().fillna(0)
        data_resampled = data_resampled.assign(position=position)
        self.results = data_resampled

    def optimize_strategy(self, freq_range, sma_s_range, sma_l_range, metric="Multiple"):
        '''
        Backtests strategy for different parameter values incl. Optimization and Reporting.

        Parameters
        ============
        freq_range: tuple
            A tuple of the form (start, end, step size) specifying the range of frequencies to be tested.

        sma_s_range: tuple
            A tuple of the form (start, end, step size) specifying the range of short moving average (SMA) window sizes to be tested.

        sma_l_range: tuple
            A tuple of the form (start, end, step size) specifying the range of long moving average (SMA) window sizes to be tested.

        metric: str (default: "Multiple")
            A performance metric to be optimized, which can be one of the following: "Multiple", "Sharpe", "Sortino", "Calmar", or "Kelly".
        '''

        # Use only Close prices
        self.data = self.data.loc[:, ["Close"]]
        # performance_function = self.perf_obj.performance_functions[metric]

        freqs = range(*freq_range)
        sma_s = range(*sma_s_range)
        sma_l = np.arange(*sma_l_range)  # NEW!!!

        combinations = list(product(freqs, sma_s, sma_l))
        # remove combinations where sma_s > sma_l
        combinations = [(_, sma_s, sma_l) for (_, sma_s, sma_l) in combinations if sma_l > sma_s]  # filter the combinations list
        # remove combinations there freq > 180 and not multiple of 30
        combinations = list(filter(lambda x: x[0] <= 180 or x[0] % 30 == 0, combinations))

        for (freq, sma_s_val, sma_l_val) in tqdm(combinations):
            self.prepare_data(freq, sma_s_val, sma_l_val)
            if metric != "Multiple":
                self.upsample()
            self.run_backtest()
            # set strategy data and calculate performance
            self.perf_obj.set_data(self.results)
            self.perf_obj.calculate_performance()
            # store strategy performance data for further plotting
            params_dic = {"freq": freq, "sma_s": sma_s_val, "sma_l": sma_l_val}
            self.dataploter.store_testcase_data(self.perf_obj, params_dic) # comb[0] is current data freq

        logger.info(f"Total number of executed tests: {len(combinations)}.")


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


if __name__ == "__main__":
    symbol = "XRPUSDT"
    start = ",2021-11-20"
    end = "2022-08-20"
    ptc = 0.00007
    ema = SMABacktester(filepath=filepath, symbol=symbol, start=start, end=end, tc=ptc)
    ema.optimize_strategy((1, 30, 10), (10, 30, 10), (10, 50, 10), metric="Calmar")