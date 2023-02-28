import os
import sys
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from utilities.performance import Performance
from utilities.data_plot import DataPlot

from backtester_base import VectorBacktesterBase

import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import threading
from utilities.logger import logger

class RSIBacktester(VectorBacktesterBase):

    def __init__(self, filepath, symbol, tc=0.00007, dataset="training", multiple_only=True, start=None, end=None):
        super().__init__(filepath=filepath, symbol=symbol, start=start, end=end, tc=tc, dataset=dataset)
        self.indicator = "RSI"
        self.perf_obj = Performance(symbol=symbol, multiple_only=multiple_only)
        self.dataploter = DataPlot(dataset, self.indicator, symbol)

    def test_strategy(self, freq=5,  periods=None, rsi_upper=None, rsi_lower=None):  # Adj!!!
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
        self.periods = periods
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower

        self.prepare_data(freq, periods, rsi_lower, rsi_upper)
        #self.upsample()
        self.run_backtest()

        #data = self.results.copy()
        #data["creturns"] = data["returns"].cumsum().apply(np.exp)
        #data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)

        #data["strategy_net"] = data.strategy - data.trades * ptc
        #data["cstrategy_net"] = data.strategy_net.cumsum().apply(np.exp)

        #self.results = data

        # store the test results in the dataframe
        # set strategy data and calculate performance
        #self.perf_obj.set_data(self.results)
        #self.perf_obj.calculate_performance()
        # store strategy performance data for further plotting
        #params_dic = {"freq": freq, "periods": periods, "rsi_lower": rsi_lower, "rsi_upper": rsi_upper}
        #self.dataploter.store_testcase_data(self.perf_obj, params_dic)  # comb[0] is current data freq

    def prepare_data(self, freq, periods, rsi_lower, rsi_upper):
        ''' Prepares the Data for Backtesting.
        '''
        freq = f"{freq}min"
        data_resampled = self.data.resample(freq).last().iloc[:-1].copy()
        data_resampled['returns'] = np.log(1 + data_resampled.Close.pct_change())
        data_resampled.dropna()
        diff = data_resampled.Close.diff()

        # Calculate MA_U and MA_D
        ma_u = pd.Series(np.where(diff > 0, diff, 0)).rolling(periods).mean()
        ma_d = pd.Series(np.where(diff < 0, -diff, 0)).rolling(periods).mean()

        # Calculate RSI
        rsi = ma_u / (ma_u + ma_d) * 100

        # Determine position
        position = np.zeros(len(data_resampled))
        position[rsi < rsi_lower] = 1
        position[rsi > rsi_upper] = 0
        position = pd.Series(position, index=data_resampled.index).ffill().fillna(0)
        self.results = data_resampled.assign(position=position).dropna()

    def optimize_strategy(self, freq_range, periods_range, rsi_lower_range, rsi_upper_range, metric="Multiple"):
        # Use only Close prices
        self.data = self.data.loc[:, ["Close"]]

        performance_function = self.perf_obj.performance_functions[metric]
        freqs = np.arange(*freq_range)
        periods = np.arange(*periods_range)
        rsi_lowers = np.arange(*rsi_lower_range)
        rsi_uppers = np.arange(*rsi_upper_range)
        combinations = list(product(freqs, periods, rsi_lowers, rsi_uppers))
        combinations = [(_, _, rsi_low, rsi_up) for (_, _, rsi_low, rsi_up) in combinations if rsi_up > rsi_low]

        logger.info(f"rsi_backtester: Optimizing of {self.indicator} for {self.symbol} using in total {len(combinations)} combinations..")
        for (freq, periods, rsi_low, rsi_up) in tqdm(combinations):
                self.prepare_data(freq, periods, rsi_low, rsi_up)
                if metric != "Multiple":
                    self.upsample()
                self.run_backtest()
                #performance.append(performance_function(self.results.strategy_net))
                # set strategy data and calculate performance
                self.perf_obj.set_data(self.results)
                self.perf_obj.calculate_performance()
                # store strategy performance data for further plotting
                params_dic = {"freq": freq, "periods": periods, "rsi_lower": rsi_low, "rsi_upper": rsi_up}
                self.dataploter.store_testcase_data(self.perf_obj, params_dic)

        #print(f"Total number of executed tests: {len(performance)} ...")

    def find_best_strategy(self):
        ''' Finds the optimal strategy (global maximum) given the parameter ranges.
        '''
        best = self.results_overview.nlargest(1, "Performance")
        freq = int(best.Freq.iloc[0])
        ema_s = int(best.EMA_S.iloc[0])
        ema_l = best.EMA_L.iloc[0]  # NEW!!!
        perf = best.Performance.iloc[0]
        print("Frequency: {} | EMA_S: {} | EMA_L: {} | {}: {}".format(freq, ema_s, ema_l, self.metric,
                                                                      round(perf, 6)))
        self.test_strategy(freq, ema_s, ema_l)


if __name__ == "__main__":
    import time
    filepath = "../../../../hist_data/XRPUSDT/XRPUSDT.parquet.gzip"
    symbol = "XRPUSDT"
    start = "2021-11-20"
    end = "2022-12-20"
    ptc = 0.00007
    freqs = (5, 120, 5)
    periods = (5, 40, 5)
    rsi_upper = (60, 100, 5)
    rsi_lower = (20, 50, 5)
    rsi = RSIBacktester(filepath=filepath, symbol=symbol, start=start, end=end, tc=ptc)
    start_t = time.time()
    rsi.optimize_strategy(freqs, periods, rsi_lower, rsi_upper, metric="Multiple")
    end_t = time.time()

    #print(f"INFO: total time took to execute with 6 threads: {round(((end_t - start_t) / 60), 2)} mins.")