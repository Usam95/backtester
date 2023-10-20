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
from utilities.plot_utils.backtesting_plotter import DataPlot
from utilities.logger import Logger
from utilities.performance import Performance

# Initialize logger
logger = Logger().get_logger()


class SoBacktester(VectorBacktesterBase):

    def __init__(self, filepath, symbol, tc=0.00007, dataset="training", start=None, end=None):
        super().__init__(filepath=filepath, symbol=symbol, start=start, end=end, tc=tc, dataset=dataset)
        self.indicator = "SO"
        self.perf_obj = Performance(symbol=symbol)
        self.dataploter = DataPlot(dataset, self.indicator, self.symbol)
        self.tc = tc

    def test_strategy(self, freq, k_period, d_period):
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

        self.generate_signals(freq, k_period, d_period)
        self.run_backtest()
        self.store_results(freq, k_period, d_period)

    def store_results(self, freq, k_period, d_period):
        # set strategy data and calculate performance
        self.perf_obj.set_data(self.results)
        self.perf_obj.calculate_performance()
        # store strategy performance data for further plotting
        params_dic = {"freq": freq, "k_period": k_period, "d_period": d_period}
        self.dataploter.store_testcase_data(self.perf_obj, params_dic)

    def generate_signals(self, freq, k_period, d_period):
        ''' Prepares the Data for Backtesting. '''
        freq = f"{freq}min"
        data_resampled = self.data.resample(freq).last().dropna().iloc[:-1].copy()
        data_resampled["returns"] = np.log(1 + data_resampled.Close.pct_change())

        # Calculate %K and %D
        high = data_resampled["High"].rolling(window=k_period).max()
        low = data_resampled["Low"].rolling(window=k_period).min()
        k_value = 100 * (data_resampled["Close"] - low) / (high - low)
        d_value = k_value.rolling(window=d_period).mean()

        # Define position based on crossover
        position = np.where(k_value > d_value, 1, 0)

        position = pd.Series(position, index=data_resampled.index).ffill().fillna(0)
        data_resampled = data_resampled.assign(position=position)
        self.results = data_resampled

    def objective(self, trial):
        freq = trial.suggest_int('freq', *self.freq_range)

        # Adjust the range for k_period to be less than d_period_range's minimum.
        k_period = trial.suggest_int('k_period', *self.k_period_range)
        d_period = trial.suggest_int('d_period', *self.d_period_range)

        self.generate_signals(freq, k_period, d_period)
        self.run_backtest()
        # store performance results for current parameter combination
        self.store_results(freq, k_period, d_period)
        performance = getattr(self.perf_obj, self.metric)

        return -performance

    def optimize_strategy(self, freq_range, k_period_range, d_period_range,
                          metric="outperf_net", opt_method="grid", bayesian_trials=100):

        self.data = self.data.loc[:, ["Close", "High", "Low"]]
        if opt_method == "grid":
            freqs = range(*freq_range)
            k_periods = range(*k_period_range)
            d_periods = np.arange(*d_period_range)

            combinations = list(product(freqs, k_periods, d_periods))
            combinations = [(_, k_periods, d_periods) for (_, k_periods, d_periods) in combinations if d_periods > k_periods]
            logger.info(
                f"Optimizing of {self.indicator} for {self.symbol} using in total {len(combinations)} combinations..")

            for (freq, k_period, d_period) in tqdm(combinations):
                self.generate_signals(freq, k_period, d_period)
                if self.metric != "outperf_net":
                    self.upsample()
                self.run_backtest()
                # store performance results for current parameter combination
                self.store_results(freq, k_period, d_period)

            logger.info(f"Total number of executed tests: {len(combinations)} ..")

        elif opt_method == "bayesian":
            self.freq_range = freq_range
            self.k_period_range = k_period_range
            self.d_period_range = d_period_range
            self.metric = metric

            study = optuna.create_study(direction='minimize')
            study.optimize(self.objective, n_trials=bayesian_trials)
            best_params = study.best_params
            best_performance = -study.best_value

            logger.info(f"Best parameters: {best_params}, Best performance: {best_performance}")
            self.dataploter.store_testcase_data(self.perf_obj, best_params)
            logger.info(
                f"Optimization completed with best parameters: {best_params} and performance: {best_performance}")

