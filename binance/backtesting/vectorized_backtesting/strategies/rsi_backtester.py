import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from utilities.Performance import Performance
from utilities.DataPlot import DataPlot

from backtester_base import VectorBacktesterBase

import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import threading
from utilities.logger import logger

class RSIBacktester(VectorBacktesterBase):

    def __init__(self, filepath, symbol, tc=0.00007, dataset="training", start=None, end=None):
        super().__init__(filepath=filepath, symbol=symbol, start=start, end=end, tc=tc, dataset=dataset)
        self.indicator = "EMA"
        self.perf_obj = Performance(symbol=symbol)
        self.dataploter = DataPlot(dataset, self.indicator, self.symbol)

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
        self.upsample()
        self.run_backtest()

        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)

        data["strategy_net"] = data.strategy - data.trades * ptc
        data["cstrategy_net"] = data.strategy_net.cumsum().apply(np.exp)

        self.results = data

        # store the test results in the dataframe
        # set strategy data and calculate performance
        self.perf_obj.set_data(self.results)
        self.perf_obj.calculate_performance()
        # store strategy performance data for further plotting
        params_dic = {"freq": freq, "periods": periods, "rsi_lower": rsi_lower, "rsi_upper": rsi_upper}
        self.dataploter.store_testcase_data(self.perf_obj, params_dic)  # comb[0] is current data freq

    def prepare_data(self, freq, periods, rsi_lower, rsi_upper):
        ''' Prepares the Data for Backtesting.
        '''
        data = self.data.Close.to_frame().copy()
        freq = "{}min".format(freq)
        resamp = data.resample(freq).last().dropna().iloc[:-1]

        # Calculate U and D
        resamp["U"] = np.where(resamp.Close.diff() > 0, resamp.Close.diff(), 0)
        resamp["D"] = np.where(resamp.Close.diff() < 0, -resamp.Close.diff(), 0)

        # Calculate MA_U and MA_D
        resamp["MA_U"] = resamp.U.rolling(periods).mean()
        resamp["MA_D"] = resamp.D.rolling(periods).mean()

        # Calculate RSI
        resamp["RSI"] = resamp.MA_U / (resamp.MA_U + resamp.MA_D) * 100

        # Determine position
        resamp["position"] = np.where(resamp.RSI > rsi_upper, -1, np.nan)
        resamp["position"] = np.where(resamp.RSI < rsi_lower, 1, resamp.position)
        resamp["position"] = resamp.position.ffill().fillna(0)

        self.results = resamp
        return resamp

    def optimize_strategy(self, freq_range, periods_range, rsi_lower_range, rsi_upper_range, metric="Multiple"):

        performance_function = self.perf_obj.performance_functions[metric]

        freqs = np.arange(*freq_range)
        periods = np.arange(*periods_range)
        rsi_lowers = np.arange(*rsi_lower_range)
        rsi_uppers = np.arange(*rsi_upper_range)
        combinations = list(product(freqs, periods, rsi_lowers, rsi_uppers))
        logger.info(f"rsi_backtester: Optimizing of {self.indicator} for {self.symbol} using in total {len(combinations)} combinations..")
        performance = []

        for comb in combinations:
            if comb[2] < comb[3]:
                try:
                    self.prepare_data(comb[0], comb[1], comb[2], comb[3])
                    self.upsample()
                    self.run_backtest()
                    #print(f"INFO: run_backtest completed: {self.results.strategy_net=} ")
                    performance.append(performance_function(self.results.strategy_net))
                    # set strategy data and calculate performance
                    self.perf_obj.set_data(self.results)
                    self.perf_obj.calculate_performance()
                    # store strategy performance data for further plotting
                    params_dic = {"freq": comb[0], "periods": comb[1], "rsi_lower": comb[2], "rsi": comb[3]}
                    self.dataploter.store_testcase_data(self.perf_obj, params_dic)
                except Exception as e:
                    print(f"ERROR: {e}")
        print(f"Total number of tests: {len(performance)}")


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
    filepath = "../../../hist_data/XRPUSDT/XRPUSDT.parquet.gzip"
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