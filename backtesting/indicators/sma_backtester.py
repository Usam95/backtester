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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from .backtester_base import VectorBacktesterBase
from utilities.data_plot import DataPlot
from utilities.logger import Logger
from utilities.performance import Performance

# Initialize logger
logger = Logger().get_logger()


class SMABacktester(VectorBacktesterBase):

    def __init__(self, filepath, symbol, tc=0.00007, dataset="training", start=None, end=None):
        super().__init__(filepath=filepath, symbol=symbol, start=start, end=end, tc=tc, dataset=dataset)
        self.indicator = "SMA"
        self.perf_obj = Performance(symbol=symbol)
        self.dataploter = DataPlot(dataset, self.indicator, self.symbol)

    def test_strategy(self, freq=5, sma_s_val=50, sma_l_val=200):
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

        self.generate_signals(freq, sma_s_val, sma_l_val)
        self.run_backtest()
        self.store_results(freq, sma_s_val, sma_l_val)

    def generate_signals(self, freq, sma_s_val, sma_l_val):  # Adj!!!
        ''' Prepares the Data for Backtesting.
        '''
        freq = f"{freq}min"
        data_resampled = self.data.resample(freq).last().dropna().iloc[:-1].copy()
        data_resampled["returns"] = np.log(1 + data_resampled.Close.pct_change())
        data_resampled.dropna()

        ######### INSERT THE STRATEGY SPECIFIC CODE HERE ##################
        sma_s = data_resampled["Close"].rolling(window=sma_s_val, min_periods=sma_s_val).mean()
        sma_l = data_resampled["Close"].rolling(window=sma_l_val, min_periods=sma_l_val).mean()
        position = np.zeros(len(data_resampled))
        position[sma_s > sma_l] = 1
        ###################################################################
        position = pd.Series(position, index=data_resampled.index).ffill().fillna(0)
        data_resampled = data_resampled.assign(position=position)
        self.results = data_resampled

    def store_results(self, freq, sma_s_val, sma_l_val):
        # set strategy data and calculate performance
        self.perf_obj.set_data(self.results)
        self.perf_obj.calculate_performance()
        # store strategy performance data for further plotting
        params_dic = {"freq": freq, "sma_s": sma_s_val, "sma_l": sma_l_val}
        self.dataploter.store_testcase_data(self.perf_obj, params_dic)

    def objective(self, trial):
        freq = trial.suggest_int('freq', *self.freq_range)
        sma_s_val = trial.suggest_int('sma_s', *self.sma_s_range)
        sma_l_val = trial.suggest_int('sma_l', *self.sma_l_range)

        if sma_l_val <= sma_s_val:
            return float('inf')

        self.generate_signals(freq, sma_s_val, sma_l_val)

        if self.metric != "outperf_net":
            self.upsample()

        self.run_backtest()

        # store performance results for current parameter combination
        self.store_results(freq, sma_s_val, sma_l_val)

        # Fetch the desired performance metric from the perf_obj instance based on the metric attribute
        performance = getattr(self.perf_obj, self.metric)

        # Return negative performance as Optuna tries to minimize the objective
        return -performance

    def optimize_strategy(self, freq_range, sma_s_range, sma_l_range,
                          metric="outperf_net", opt_method="grid", bayesian_trials=100):
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

        if opt_method == "grid":
            # Grid search optimization

            freqs = range(*freq_range)
            sma_s = range(*sma_s_range)
            sma_l = np.arange(*sma_l_range)

            combinations = list(product(freqs, sma_s, sma_l))
            combinations = [(_, sma_s, sma_l) for (_, sma_s, sma_l) in combinations if sma_l > sma_s]
            logger.info(
                f"ema_backtester: Optimizing of {self.indicator} for {self.symbol} using in total {len(combinations)} combinations..")

            for (freq, sma_s_val, sma_l_val) in tqdm(combinations):
                self.generate_signals(freq, sma_s_val, sma_l_val)
                if self.metric != "outperf_net":
                    self.upsample()
                self.run_backtest()
                # store performance results for current parameter combination
                self.store_results(freq, sma_s_val, sma_l_val)
            logger.info(f"Total number of executed tests: {len(combinations)} ..")

        elif opt_method == "bayesian":

            # Bayesian optimization
            self.freq_range = freq_range
            self.sma_s_range = sma_s_range
            self.sma_l_range = sma_l_range
            self.metric = metric

            study = optuna.create_study(direction='minimize')
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
        sma_s = int(best.SMA_S.iloc[0])
        sma_l = best.SMA_L.iloc[0]  # NEW!!!
        perf = best.Performance.iloc[0]
        print("Frequency: {} | EMA_S: {} | EMA_L: {} | {}: {}".format(freq, sma_s, sma_l, self.metric,
                                                                      round(perf, 6)))
        self.test_strategy(freq, sma_s, sma_l)


if __name__ == "__main__":
    filepath = "../../hist_data/XRPUSDT/XRPUSDT.parquet.gzip"
    symbol = "XRPUSDT"
    start = "2022-06-20"
    end = "2022-08-20"
    ptc = 0.00007
    sma = SMABacktester(filepath=filepath, symbol=symbol, start=None, end=end, tc=ptc)
    sma.optimize_strategy((5, 10, 5), (10, 20, 5), (10, 40, 5), metric="outperf_net",  opt_method="grid")
    sma.dataploter.store_data("../")