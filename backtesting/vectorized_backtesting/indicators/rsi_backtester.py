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


class RsiBacktester(VectorBacktesterBase):

    def __init__(self, filepath, symbol, tc=0.00007, dataset="training", multiple_only=True, start=None, end=None):
        super().__init__(filepath=filepath, symbol=symbol, start=start, end=end, tc=tc, dataset=dataset)
        self.indicator = "RSI"
        self.perf_obj = Performance(symbol=symbol, multiple_only=multiple_only)
        self.dataploter = DataPlot(dataset, self.indicator, symbol)

    def test_strategy(self, freq=5,  periods=None, rsi_lower=None, rsi_upper=None):  # Adj!!!
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

        self.generate_signals(freq, periods, rsi_lower, rsi_upper)
        self.run_backtest()
        self.store_results(freq, periods, rsi_lower, rsi_upper)

    def generate_signals(self, freq, periods, rsi_lower, rsi_upper):
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

    def store_results(self, freq, periods, rsi_low, rsi_up):
        # set strategy data and calculate performance
        self.perf_obj.set_data(self.results)
        self.perf_obj.calculate_performance()
        # store strategy performance data for further plotting
        params_dic = {"freq": freq, "periods": periods, "rsi_lower": rsi_low, "rsi_upper": rsi_up}
        self.dataploter.store_testcase_data(self.perf_obj, params_dic)

    def objective(self, trial):
        freq = trial.suggest_int('freq', *self.freq_range)
        periods = trial.suggest_int('periods', *self.periods_range)
        rsi_lower = trial.suggest_int('rsi_lower', *self.rsi_lower_range)
        rsi_upper = trial.suggest_int('rsi_upper', *self.rsi_upper_range)

        if rsi_upper <= rsi_lower:
            return float('inf')

        self.generate_signals(freq, periods, rsi_lower, rsi_upper)

        if self.metric != "Multiple":
            self.upsample()

        self.run_backtest()
        # store performance results for current parameter combination
        self.store_results(freq, periods, rsi_lower, rsi_upper)

        # Fetch the desired performance metric from the perf_obj instance based on the metric attribute
        performance = getattr(self.perf_obj, self.metric)

        # Return negative performance as Optuna tries to minimize the objective
        return -performance

    def optimize_strategy(self, freq_range, periods_range, rsi_lower_range, rsi_upper_range,
                          metric="outperf_net", opt_method="grid", bayesian_trials=100):
        self.data = self.data.loc[:, ["Close"]]
        self.metric = metric

        if opt_method == "grid":
            freqs = np.arange(*freq_range)
            periods = np.arange(*periods_range)
            rsi_lowers = np.arange(*rsi_lower_range)
            rsi_uppers = np.arange(*rsi_upper_range)

            combinations = list(product(freqs, periods, rsi_lowers, rsi_uppers))
            combinations = [(_, _, rsi_low, rsi_up) for (_, _, rsi_low, rsi_up) in combinations if rsi_up > rsi_low]
            logger.info(
                f"RSIBacktester: Optimizing of {self.indicator} for {self.symbol} using in total {len(combinations)} combinations..")

            for (freq, periods, rsi_low, rsi_up) in tqdm(combinations):
                self.generate_signals(freq, periods, rsi_low, rsi_up)
                if metric != "Multiple":
                    self.upsample()
                self.run_backtest()
                # store performance results for current parameter combination
                self.store_results(freq, periods, rsi_low, rsi_up)

            logger.info(f"Total number of executed tests: {len(combinations)}.")

        elif opt_method == "bayesian":
            self.freq_range = freq_range
            self.periods_range = periods_range
            self.rsi_lower_range = rsi_lower_range
            self.rsi_upper_range = rsi_upper_range

            study = optuna.create_study(direction='minimize')
            study.optimize(self.objective, n_trials=bayesian_trials)  # Adjust n_trials as needed

            best_params = study.best_params
            best_performance = -study.best_value

            logger.info(f"Best parameters: {best_params}, Best performance: {best_performance}")
            self.dataploter.store_testcase_data(self.perf_obj, best_params)
            logger.info(
                f"Optimization completed with best parameters: {best_params} and performance: {best_performance}")

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
    filepath = "../../hist_data/XRPUSDT//XRPUSDT.parquet.gzip"
    symbol = "XRPUSDT"
    start = "2021-11-20"
    end = "2022-12-20"
    ptc = 0.00007
    freqs = (5, 10, 5)
    periods = (5, 40, 5)
    rsi_upper = (60, 100, 5)
    rsi_lower = (20, 50, 5)
    rsi = RsiBacktester(filepath=filepath, symbol=symbol, start=start, end=end, tc=ptc)
    start_t = time.time()
    rsi.optimize_strategy(freqs, periods, rsi_lower, rsi_upper, metric="strategy_multiple_net", opt_method="grid")
    end_t = time.time()
    rsi.dataploter.store_data("../")