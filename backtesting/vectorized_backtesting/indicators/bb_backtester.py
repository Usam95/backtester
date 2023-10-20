# Standard libraries
import os
import sys
from itertools import product

# Third-party libraries
import numpy as np
import pandas as pd
import optuna
from tqdm import tqdm

# Application/Library specific imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..', '..')))

from .backtester_base import VectorBacktesterBase
from utilities.plot_utils.backtesting_plotter import DataPlot
from utilities.logger import Logger
from utilities.performance import Performance

# Initialize logger
logger = Logger().get_logger()


class BbBacktester(VectorBacktesterBase):

        def __init__(self, filepath, symbol, tc=0.00007, dataset="training", start=None, end=None):
            super().__init__(filepath=filepath, symbol=symbol, start=start, end=end, tc=tc)
            self.indicator = "BB"
            self.perf_obj = Performance()
            self.dataploter = DataPlot(dataset, self.indicator, self.symbol)

        def test_strategy(self, freq, window , dev):
            """
            Prepares the data and backtests the trading strategy incl. reporting (Wrapper).

            Parameters
            ============
            freq: int
                data frequency/granularity to work with (in minutes)

            window: int
                time window (number of bars) to calculate the simple moving average price (SMA).

            dev: int
                number of standard deviations to calculate upper and lower bands.
            """

            self.generate_signals(freq, window, dev)
            self.run_backtest()
            self.store_results(freq, window, dev)

        def generate_signals(self, freq, window, dev):  # Adj!!!
            ''' Prepares the Data for Backtesting.
            '''
            freq = "{}min".format(freq)
            data_resampled = self.data.copy().resample(freq).last().dropna().iloc[:-1]
            data_resampled["returns"] = np.log(1 + data_resampled.Close.pct_change())
            data_resampled.dropna(inplace=True)

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

        def store_results(self, freq, window, dev):
            # set strategy data and calculate performance
            self.perf_obj.set_data(self.results)
            self.perf_obj.calculate_performance()
            # store strategy performance data for further plotting
            params_dic = {"freq": freq, "window": window, "dev": dev}
            self.dataploter.store_testcase_data(self.perf_obj, params_dic)  # comb[0] is current data freq

        def objective(self, trial):
            freq = trial.suggest_int('freq', *self.freq_range)
            window = trial.suggest_int('window', *self.window_range)
            dev = trial.suggest_int('dev', *self.dev_range)

            self.generate_signals(freq, window, dev)

            if self.metric != "outperf_net":
                self.upsample()

            self.run_backtest()

            # store performance results for current parameter combination
            self.store_results(freq, window, dev)

            # Fetch the desired performance metric from the perf_obj instance based on the metric attribute
            performance = getattr(self.perf_obj, self.metric)

            # Return negative performance as Optuna tries to minimize the objective
            return -performance

        def optimize_strategy(self, freq_range, window_range, dev_range,
                              metric="outperf_net", opt_method="grid", bayesian_trials=100):
            # Use only Close prices
            self.data = self.data.loc[:, ["Close"]]
            self.metric = metric

            if opt_method == "grid":
                # Grid search optimization
                freqs = range(*freq_range)
                windows = range(*window_range)
                devs = np.arange(*dev_range)

                combinations = list(product(freqs, windows, devs))
                logger.info(
                    f"BBBacktester: Optimizing of {self.indicator} for {self.symbol} using in total {len(combinations)} combinations..")

                for (freq, window, dev) in tqdm(combinations):
                    self.generate_signals(freq, window, dev)
                    if metric != "outperf_net":
                        self.upsample()
                    self.run_backtest()
                    # store performance results for current parameter combination
                    self.store_results(freq, window, dev)
                logger.info(f"Total number of executed tests: {len(combinations)}.")

            elif opt_method == "bayesian":
                # Bayesian optimization
                self.freq_range = freq_range
                self.window_range = window_range
                self.dev_range = dev_range
                self.metric = metric

                study = optuna.create_study(direction='minimize')
                print(f"{bayesian_trials=}")
                study.optimize(self.objective, n_trials=bayesian_trials)  # Set n_trials as desired

                best_params = study.best_params
                best_performance = -study.best_value

                logger.info(f"Best parameters: {best_params}, Best performance: {best_performance}")
                # Optionally store and visualize results
                self.dataploter.store_testcase_data(self.perf_obj, best_params)
                logger.info(
                    f"Optimization completed with best parameters: {best_params} and performance: {best_performance}")

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
    filepath = "../../hist_data/XRPUSDT/XRPUSDT.parquet.gzip"
    symbol = "XRPUSDT"
    start = "2018-05-01"
    end = "2022-12-31"
    ptc = 0.000750
    bb = BbBacktester(filepath=filepath, symbol=symbol, start=start, end=end, tc=ptc)
    bb.test_strategy(freq=210, window=105, dev=4)
   # bb.optimize_strategy((<1>, 30, 10), (10, 30, 10), (10, 50, 10), metric="Calmar")
    #print(len(bb.dataploter.df))
    #print(bb.dataploter.df.head())