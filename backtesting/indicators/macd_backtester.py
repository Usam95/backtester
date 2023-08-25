import optuna
import numpy as np
from tqdm import tqdm
import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utilities.data_plot import DataPlot
from utilities.performance import Performance
from itertools import product
from backtester_base import VectorBacktesterBase

from utilities.logger import Logger
logger = Logger().get_logger()



class MACDBacktester(VectorBacktesterBase):
    ''' Class for the vectorized backtesting of MACD-based trading strategies.

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    EMA_S: int
        time window in days for shorter EMA
    EMA_L: int
        time window in days for longer EMA
    signal_mw: int
        time window is days for MACD Signal
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    tc: float
        proportional transaction costs per trade


    Methods
    =======
    get_data:
        retrieves and prepares the data

    set_parameters:
        sets new MACD parameter(s)

    test_strategy:
        runs the backtest for the MACD-based strategy

    plot_results:
        plots the performance of the strategy compared to buy and hold

    update_and_run:
        updates MACD parameters and returns the negative absolute performance (for minimization algorithm)

    optimize_parameters:
        implements a brute force optimization for the three MACD parameters
    '''

    def __init__(self, filepath, symbol, tc=0.00007, dataset="training", start=None, end=None):
        super().__init__(filepath=filepath, symbol=symbol, start=start, end=end, tc=tc, dataset=dataset)
        self.indicator = "MACD"
        self.perf_obj = Performance(symbol=symbol)
        self.dataploter = DataPlot(dataset, self.indicator, self.symbol)

    def __repr__(self):
        return "MACDBacktester(symbol = {}, MACD({}, {}, {}), start = {}, end = {})".format(self.symbol, self.EMA_S,
                                                                                            self.EMA_L, self.signal_mw,
                                                                                            self.start, self.end)


    def test_strategy(self, freq=5, ema_s=50, ema_l=200, signal_mw=9):  # Adj!!!
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
        self.prepare_data(freq, ema_s, ema_l, signal_mw)
        self.upsample()
        self.run_backtest()

        data = self.results.copy()
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data

        # store the test results in the dataframe
        # set strategy data and calculate performance
        self.perf_obj.set_data(self.results)
        self.perf_obj.calculate_performance()
        # store strategy performance data for further plotting
        params_dic = {"freq": freq, "ema_s": ema_s, "ema_l": ema_l, "signal_mw": signal_mw}
        self.dataploter.store_testcase_data(self.perf_obj, params_dic)  # comb[0] is current data freq

    def prepare_data(self, freq, ema_s_val, ema_l_val, signal_mw):  # Adj!!!
        ''' Prepares the Data for Backtesting.
        '''
        freq = f"{freq}min"
        print(f"Len of self.data: {len(self.data)}")
        data_resampled = self.data.resample(freq).last().dropna().iloc[:-1].copy()
        print(f"Len of data_resampled 1: {len(data_resampled)}")
        data_resampled["returns"] = np.log(1 + data_resampled.Close.pct_change())
        data_resampled.dropna()
        ######### INSERT THE STRATEGY SPECIFIC CODE HERE ##################

        ema_s = data_resampled["Close"].ewm(span=ema_s_val - 1, min_periods=ema_s_val).mean()
        ema_l = data_resampled["Close"].ewm(span=ema_l_val - 1, min_periods=ema_l_val).mean()
        macd = (ema_s - ema_l)
        macd_signal = macd.ewm(span=signal_mw, min_periods=signal_mw).mean()

        position = np.zeros(len(data_resampled))
        position[macd > macd_signal] = 1

        ###################################################################
        position = pd.Series(position, index=data_resampled.index).ffill().fillna(0)
        data_resampled = data_resampled.assign(position=position).dropna()
        self.results = data_resampled
        print(f"Len of data_resampled 2: {len(data_resampled)}")
        print(f"Len of self.results: {len(self.results)}")

    def objective(self, trial):
        freq = trial.suggest_int('freq', *self.freq_range)
        ema_s_val = trial.suggest_int('ema_s', *self.ema_s_range)
        ema_l_val = trial.suggest_int('ema_l', *self.ema_l_range)
        signal_mw = trial.suggest_int('signal_mw', *self.signal_mw_range)

        if ema_l_val <= ema_s_val:
            return float('inf')

        self.prepare_data(freq, ema_s_val, ema_l_val, signal_mw)
        self.run_backtest()

        # set strategy data and calculate performance
        self.perf_obj.set_data(self.results)
        self.perf_obj.calculate_performance()

        # Fetch the desired performance metric from the perf_obj instance based on the metric attribute
        performance = getattr(self.perf_obj, self.metric)

        # Return negative performance as Optuna tries to minimize the objective
        return -performance

    def optimize_strategy(self, freq_range, ema_s_range, ema_l_range, signal_mw_range, metric="Multiple",
                          opt_method="grid"):
        '''Backtests strategy for different parameter values incl. Optimization and Reporting (Wrapper).

        Parameters
        ==========
        freq_range: tuple
            tuples of the form (start, end, step size).

        ema_s_range: tuple
            tuples of the form (start, end, step size).

        ema_l_range: tuple
            tuples of the form (start, end, step size).

        signal_mw_range: tuple
            tuples of the form (start, end, step size).

        metric: str
            performance metric to be optimized (can be: "Multiple", "Sharpe", "Sortino", "Calmar", "Kelly")

        opt_method: str
            optimization method (can be: "grid" or "bayesian")
        '''

        self.data = self.data.loc[:, ["Close"]]
        self.metric = metric

        if opt_method == "grid":
            freqs = range(*freq_range)
            ema_s_vals = range(*ema_s_range)
            ema_l_vals = range(*ema_l_range)
            signal_mws = range(*signal_mw_range)
            ema_pairs = [(ema_s, ema_l) for ema_s in ema_s_vals for ema_l in ema_l_vals if ema_l > ema_s]
            combinations = list(product(freqs, ema_pairs, signal_mws))

            logger.info(
                f"macd_backtester: Optimizing of {self.indicator} for {self.symbol} using in total {len(combinations)} combinations..")

            for (freq, (ema_s, ema_l), signal_mw) in tqdm(combinations):
                self.prepare_data(freq, ema_s, ema_l, signal_mw)
                if metric != "Multiple":
                    self.upsample()
                self.run_backtest()
                self.perf_obj.set_data(self.results)
                self.perf_obj.calculate_performance()
                params_dic = {"freq": freq, "ema_s": ema_s, "ema_l": ema_l, "Signal_Mw": signal_mw}
                self.dataploter.store_testcase_data(self.perf_obj, params_dic)

            logger.info(f"Total number of executed tests: {len(combinations)}.")

        elif opt_method == "bayesian":
            self.freq_range = freq_range
            self.ema_s_range = ema_s_range
            self.ema_l_range = ema_l_range
            self.signal_mw_range = signal_mw_range

            study = optuna.create_study(direction='minimize')
            study.optimize(self.objective, n_trials=1000)  # Set n_trials as desired

            best_params = study.best_params
            best_performance = -study.best_value

            logger.info(f"Best parameters: {best_params}, Best performance: {best_performance}")
            self.dataploter.store_testcase_data(self.perf_obj, best_params)
            logger.info(
                f"Optimization completed with best parameters: {best_params} and performance: {best_performance}")

        else:
            logger.error("Unknown optimization method. Please choose 'grid' or 'bayesian'.")

    def find_best_strategy(self):
        ''' Finds the optimal strategy (global maximum) given the parameter ranges.
        '''
        best = self.results_overview.nlargest(1, "Performance")
        freq = int(best.freq.iloc[0])
        sma_s = int(best.sma_s.iloc[0])
        sma_l = best.sma_l.iloc[0]  # NEW!!!
        perf = best.Performance.iloc[0]
        logger.info("freq: {} | ema_s: {} | ema_L: {} | {}: {}".format(freq, sma_s, sma_l,
                                                                       self.metric, round(perf, 6)))
        self.test_strategy(freq, sma_s, sma_l)


if __name__ == "__main__":
    filepath = "../../hist_data/XRPUSDT//XRPUSDT.parquet.gzip"
    symbol = "XRPUSDT"
    start = "2018-06-20"
    end = "2022-08-20"
    ptc = 0.01
    macd = MACDBacktester(filepath=filepath, symbol=symbol, start=start, end=end, tc=ptc)
    macd.optimize_strategy((5, 10, 5), (10, 20, 5), (10, 40, 5), (10, 40, 5), metric="outperf_net",  opt_method="bayesian")
    macd.dataploter.store_data("../")