
from utilities.performance import Performance
from utilities.data_plot import DataPlot
from backtester_base import VectorBacktesterBase
from tqdm import tqdm
import numpy as np
from itertools import product
import pandas as pd

from utilities.logger import Logger
logger = Logger().get_logger()


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
            freq = "{}min".format(freq)
            data_resampled = self.data.copy().resample(freq).last().dropna().iloc[:-1]
            data_resampled["returns"] = np.log(1 + data_resampled.Close.pct_change())
            data_resampled.dropna()

            # Calculate the rolling mean and standard deviation
            rolling_mean = data_resampled["Close"].rolling(window).mean()
            rolling_std = data_resampled["Close"].rolling(window).std()

            # Calculate upper and lower bands
            upper_band = rolling_mean + rolling_std * dev
            lower_band = rolling_mean - rolling_std * dev

            # Calculate distance and position
            position = np.where(data_resampled.Close < lower_band, 1, np.nan)
            position = np.where(data_resampled.Close > upper_band, 0, position)
            position = pd.Series(position, index=data_resampled.index).ffill().fillna(0)
            # Create a new dataframe with required columns
            data_resampled = data_resampled.assign(position=position)
            data_resampled.dropna(inplace=True)
            self.results = data_resampled

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

            self.data = self.data.loc[:, ["Close"]]
            # performance_function = self.perf_obj.performance_functions[metric]

            freqs = range(*freq_range)
            windows = range(*window_range)
            devs = np.arange(*dev_range)  # NEW!!!
            combinations = list(product(freqs, windows, devs))

            for (freq, window, dev) in tqdm(combinations):
                self.prepare_data(freq, window, dev)
                if metric != "Multiple":
                    self.upsample()
                self.run_backtest()
                #performance.append(performance_function(self.results.strategy))
                # set strategy data and calculate performance
                self.perf_obj.set_data(self.results)
                self.perf_obj.calculate_performance()
                # store strategy performance data for further plotting
                params_dic = {"freq": freq, "window": window, "stddev": dev}
                self.dataploter.store_testcase_data(self.perf_obj, params_dic)  # comb[0] is current data freq
            logger.info(f"Total number of executed tests: {len(combinations)}.")

        def find_best_strategy(self):
            ''' Finds the optimal strategy (global maximum) given the parameter ranges.
            '''
            best = self.results_overview.nlargest(1, "Performance")
            freq = int(best.Freq.iloc[0])
            sma = int(best.Windows.iloc[0])
            dev = best.Devs.iloc[0]  # NEW!!!
            perf = best.Performance.iloc[0]
            print("Frequency: {} | SMA: {} | Dev: {} | {}: {}".format(freq, sma, dev, self.metric, round(perf, 6)))
            self.test_strategy(freq, sma, dev)

if __name__ == "__main__":
    filepath = "../../../../hist_data/XRPUSDT/train/XRPUSDT.csv"
    symbol = "XRPUSDT"
    start = "2020-08-20"
    end = "2020-11-20"
    ptc = 0.00007
    #bb = BBStrategy(filepath=filepath, symbol=symbol, start=start, end=end, tc=ptc)
   # bb.optimize_strategy((1, 30, 10), (10, 30, 10), (10, 50, 10), metric="Calmar")
    #print(len(bb.dataploter.df))
    #print(bb.dataploter.df.head())