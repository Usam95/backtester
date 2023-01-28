import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from utilities.Performance import Performance
from utilities.DataPlot import DataPlot

from BacktesterBase import VectorBacktesterBase

import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

class EMABacktester(VectorBacktesterBase):

    def __init__(self, filepath, symbol, tc=0.00007, dataset="training", start=None, end=None):
        super().__init__(filepath=filepath, symbol=symbol, start=start, end=end, tc=tc, dataset=dataset)
        self.indicator = "EMA"
        self.perf_obj = Performance(symbol=symbol)
        self.dataploter = DataPlot(dataset, self.indicator, self.symbol)

    def test_strategy(self, freq=5, EMA_S=50, EMA_L=200):  # Adj!!!
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
        self.EMA_S = EMA_S
        self.EMA_L = EMA_L  # NEW!!!

        self.prepare_data(freq, EMA_S, EMA_L)
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
        params_dic = {"freq": freq, "EMA_S": EMA_S, "EMA_L": EMA_L}
        self.dataploter.store_testcase_data(self.perf_obj, params_dic)  # comb[0] is current data freq
        #self.print_performance()

    def prepare_data(self, freq, EMA_S, EMA_L):  # Adj!!!
        ''' Prepares the Data for Backtesting.
        '''
        data = self.data.Close.to_frame().copy()
        freq = "{}min".format(freq)
        resamp = data.resample(freq).last().dropna().iloc[:-1]

        ######### INSERT THE STRATEGY SPECIFIC CODE HERE ##################

        resamp["EMA_S"] = resamp["Close"].ewm(span=EMA_S, min_periods=EMA_S).mean()
        resamp["EMA_L"] = resamp["Close"].ewm(span=EMA_L, min_periods=EMA_L).mean()

        resamp["position"] = np.where(resamp["EMA_S"] > resamp["EMA_L"], 1, 0)

        resamp["position"] = resamp.position.ffill().fillna(0)
        ###################################################################

        resamp.dropna(inplace=True)
        self.results = resamp
        return resamp

    def optimize_strategy(self, freq_range, EMA_S_range, EMA_L_range, metric="Multiple"):  # Adj!!!
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

        self.metric = metric

        if metric == "Multiple":
            performance_function = self.perf_obj.calculate_multiple
        elif metric == "Sharpe":
            performance_function = self.perf_obj.calculate_sharpe
        elif metric == "Sortino":
            performance_function = self.perf_obj.calculate_sortino
        elif metric == "Calmar":
            performance_function = self.perf_obj.calculate_calmar
        elif metric == "Kelly":
            performance_function = self.perf_obj.calculate_kelly_criterion

        freqs = range(*freq_range)
        ema_s = range(*EMA_S_range)
        ema_l = np.arange(*EMA_L_range)  # NEW!!!

        combinations = list(product(freqs, ema_s, ema_l))
        print(f"INFO: Optimizing of {self.indicator} for {self.symbol} using in total {len(combinations)} combinations..")
        performance = []

        # for data plotting
        #self.strategy_df = pd.DataFrame()
        count = 0
        for comb in combinations:
            if comb[1] < comb[2]:
                self.prepare_data(comb[0], comb[1], comb[2])
                self.upsample()
                self.run_backtest()
                performance.append(performance_function(self.results.strategy_net))
                # set strategy data and calculate performance
                self.perf_obj.set_data(self.results)
                self.perf_obj.calculate_performance()
                # store strategy performance data for further plotting
                params_dic = {"freq":comb[0], "EMA_S":comb[1], "EMA_L":comb[2]}
                self.dataploter.store_testcase_data(self.perf_obj, params_dic) # comb[0] is current data freq
                count +=1
        print(f"Total number of test: {count}")

        #self.results_overview = pd.DataFrame(data=np.array(combinations),
        #                                     columns=["Freq", "EMA_S", "EMA_L"])
        #self.results_overview["Performance"] = performance
        #self.find_best_strategy()

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
    filepath = "../../../hist_data/XRPUSDT/test/XRPUSDT.csv"
    symbol = "XRPUSDT"
    start = ",2021-11-20"
    end = "2022-08-20"
    ptc = 0.00007
    ema = EMABacktester(filepath=filepath, symbol=symbol, start=start, end=end, tc=ptc)
    ema.optimize_strategy((1, 30, 10), (10, 30, 10), (10, 50, 10), metric="Calmar")