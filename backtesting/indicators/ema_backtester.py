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


class EMABacktester(VectorBacktesterBase):

    def __init__(self, filepath, symbol, tc=0.00007, dataset="training", start=None, end=None):
        super().__init__(filepath=filepath, symbol=symbol, start=start, end=end, tc=tc, dataset=dataset)
        self.indicator = "EMA"
        self.perf_obj = Performance(symbol=symbol)
        self.dataploter = DataPlot(dataset, self.indicator, self.symbol)

    def test_strategy(self, freq=5, ema_s_val=50, ema_l_val=200):
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

        self.generates_signals(freq, ema_s_val, ema_l_val)
        self.run_backtest()
        self.store_results(freq, ema_s_val, ema_l_val)

    def generates_signals(self, freq, ema_s_val, ema_l_val):  # Adj!!!
        ''' Prepares the Data for Backtesting.
        '''
        freq = f"{freq}min"
        data_resampled = self.data.resample(freq).last().dropna().iloc[:-1].copy()
        data_resampled["returns"] = np.log(1 + data_resampled.Close.pct_change())
        data_resampled.dropna()
        ######### INSERT THE STRATEGY SPECIFIC CODE HERE ##################

        ema_s = data_resampled["Close"].ewm(span=ema_s_val-1, min_periods=ema_s_val).mean()
        ema_l = data_resampled["Close"].ewm(span=ema_l_val-1, min_periods=ema_l_val).mean()

        position = np.zeros(len(data_resampled))
        position[ema_s > ema_l] = 1
        ###################################################################
        position = pd.Series(position, index=data_resampled.index).ffill().fillna(0)
        data_resampled = data_resampled.assign(position=position)
        self.results = data_resampled

    def store_results(self, freq, ema_s_val, ema_l_val):
        # set strategy data and calculate performance
        self.perf_obj.set_data(self.results)
        self.perf_obj.calculate_performance()
        # store strategy performance data for further plotting
        params_dic = {"freq": freq, "ema_s": ema_s_val, "ema_l": ema_l_val}
        self.dataploter.store_testcase_data(self.perf_obj, params_dic)  # comb[0] is current data freq

    def objective(self, trial):
        freq = trial.suggest_int('freq', *self.freq_range)
        ema_s_val = trial.suggest_int('ema_s', *self.ema_s_range)
        ema_l_val = trial.suggest_int('ema_l', *self.ema_l_range)

        if ema_l_val <= ema_s_val:
            return float('inf')

        self.generates_signals(freq, ema_s_val, ema_l_val)

        if self.metric != "outperf_net":
            self.upsample()

        self.run_backtest()
        # store performance results for current parameter combination
        self.store_results(freq, ema_s_val, ema_l_val)

        # Fetch the desired performance metric from the perf_obj instance based on the metric attribute
        performance = getattr(self.perf_obj, self.metric)

        # Return negative performance as Optuna tries to minimize the objective
        return -performance

    def optimize_strategy(self, freq_range, ema_s_range, ema_l_range,
                          metric="outperf_net", opt_method="grid", bayesian_trials=100):
        # Use only Close prices
        self.metric = metric
        self.data = self.data.loc[:, ["Close"]]
        if opt_method == "grid":
            # Grid search optimization

            freqs = range(*freq_range)
            ema_s = range(*ema_s_range)
            ema_l = np.arange(*ema_l_range)

            combinations = list(product(freqs, ema_s, ema_l))
            combinations = [(_, ema_s, ema_l) for (_, ema_s, ema_l) in combinations if ema_l > ema_s]
            logger.info(f"ema_backtester: Optimizing of {self.indicator} for {self.symbol} using in total {len(combinations)} combinations..")

            for (freq, ema_s_val, ema_l_val) in tqdm(combinations):
                self.generates_signals(freq, ema_s_val, ema_l_val)
                if self.metric != "outperf_net":
                    self.upsample()
                self.run_backtest()
                # store performance results for current parameter combination
                self.store_results(freq, ema_s_val, ema_l_val)

            logger.info(f"Total number of executed tests: {len(combinations)} ..")

        elif opt_method == "bayesian":

            # Bayesian optimization
            self.freq_range = freq_range
            self.ema_s_range = ema_s_range
            self.ema_l_range = ema_l_range
            self.metric = metric

            study = optuna.create_study(direction='minimize')
            study.optimize(self.objective, n_trials=bayesian_trials)  # Set n_trials as desired
            # Save ranges and metric for use in objective function
            best_params = study.best_params
            best_performance = -study.best_value

            logger.info(f"Best parameters: {best_params}, Best performance: {best_performance}")
            # Optionally store and visualize results
            self.dataploter.store_testcase_data(self.perf_obj, best_params)
            logger.info(f"Optimization completed with best parameters: {best_params} and performance: {best_performance}")

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
    filepath = "../../hist_data/XRPUSDT//XRPUSDT.parquet.gzip"
    symbol = "XRPUSDT"
    start = "2022-06-20"
    end = "2022-08-20"
    ptc = 0.01
    ema = EMABacktester(filepath=filepath, symbol=symbol, start=start, end=end, tc=ptc)
    ema.optimize_strategy((5, 100, 5), (10, 500, 4), (15, 80, 5), metric="outperf_net",  opt_method="bayesian")
    ema.dataploter.store_data("../")
