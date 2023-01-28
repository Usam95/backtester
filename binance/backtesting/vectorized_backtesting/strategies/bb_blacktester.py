import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from utilities.Performance import Performance
from utilities.DataPlot import DataPlot
from BacktesterBase import VectorBacktesterBase

import numpy as np
from itertools import product


class BBBacktester(VectorBacktesterBase):

        def __init__(self, filepath, symbol, tc=0.00007, dataset="training", start=None, end=None):
            super().__init__(filepath=filepath, symbol=symbol, start=start, end=end, tc=tc)
            self.indicator = "BB"
            self.perf_obj = Performance()
            self.dataploter = DataPlot(dataset, self.indicator, self.symbol)
        def test_strategy(self, freq=60, window=50, dev=2):  # Adj!!!
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
            self.window = window
            self.dev = dev  # NEW!!!

            self.prepare_data(freq, window, dev)
            self.upsample()
            self.run_backtest()

            data = self.results.copy()
            data["creturns"] = data["returns"].cumsum().apply(np.exp)
            data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)

            data["strategy_net"] = data.strategy - data.trades * ptc
            data["cstrategy_net"] = data.strategy_net.cumsum().apply(np.exp)
            self.results = data

        def prepare_data(self, freq, window, dev):  # Adj!!!
            ''' Prepares the Data for Backtesting.
            '''
            data = self.data.Close.to_frame().copy()
            freq = "{}min".format(freq)
            resamp = data.resample(freq).last().dropna().iloc[:-1]

            ######### INSERT THE STRATEGY SPECIFIC CODE HERE ##################
            resamp["SMA"] = resamp["Close"].rolling(window).mean()
            resamp["Lower"] = resamp["SMA"] - resamp["Close"].rolling(window).std() * dev
            resamp["Upper"] = resamp["SMA"] + resamp["Close"].rolling(window).std() * dev

            resamp["distance"] = resamp.Close - resamp.SMA
            resamp["position"] = np.where(resamp.Close < resamp.Lower, 1, np.nan)
            resamp["position"] = np.where(resamp.Close > resamp.Upper, 0, resamp["position"])
            #resamp["position"] = np.where(resamp.distance * resamp.distance.shift(1) < 0, 0, resamp["position"])
            resamp["position"] = resamp.position.ffill().fillna(0)
            ###################################################################

            resamp.dropna(inplace=True)
            self.results = resamp

            return resamp

        def optimize_strategy(self, freq_range, window_range, dev_range, metric="Multiple"):  # Adj!!!
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
            windows = range(*window_range)
            devs = np.arange(*dev_range)  # NEW!!!

            combinations = list(product(freqs, windows, devs))

            print(f"INFO: Optimizing of {self.indicator} for {self.symbol} using in total {len(combinations)} combinations..")

            performance = []
            for comb in combinations:
                self.prepare_data(comb[0], comb[1], comb[2])
                self.upsample()
                self.run_backtest()
                performance.append(performance_function(self.results.strategy))
                # set strategy data and calculate performance
                self.perf_obj.set_data(self.results)
                self.perf_obj.calculate_performance()
                # store strategy performance data for further plotting
                params_dic = {"freq": comb[0], "SMA": comb[1], "Dev": comb[2]}

                self.dataploter.store_testcase_data(self.perf_obj, params_dic)  # comb[0] is current data freq

            #self.results_overview = pd.DataFrame(data=np.array(combinations),
            #                                    columns=["Freq", "Windows", "Devs"])
            #self.results_overview["Performance"] = performance
            #self.find_best_strategy()


        def find_best_strategy(self):
            ''' Finds the optimal strategy (global maximum) given the parameter ranges.
            '''
            best = self.results_overview.nlargest(1, "Performance")
            freq = int(best.Freq.iloc[0])
            sma = int(best.Windows.iloc[0])
            dev = best.Devs.iloc[0]  # NEW!!!
            perf = best.Performance.iloc[0]
            print("Frequency: {} | SMA: {} | Dev: {} | {}: {}".format(freq, sma, dev, self.metric,
                                                                          round(perf, 6)))
            self.test_strategy(freq, sma, dev)

if __name__ == "__main__":
    filepath = "../../../hist_data/XRPUSDT/train/XRPUSDT.csv"
    symbol = "XRPUSDT"
    start = "2020-08-20"
    end = "2020-11-20"
    ptc = 0.00007
    #bb = BBStrategy(filepath=filepath, symbol=symbol, start=start, end=end, tc=ptc)
   # bb.optimize_strategy((1, 30, 10), (10, 30, 10), (10, 50, 10), metric="Calmar")
    print(len(bb.dataploter.df))
    print(bb.dataploter.df.head())