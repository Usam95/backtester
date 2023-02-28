import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from backtester_base import VectorBacktesterBase
import numpy as np
from itertools import product
from utilities.data_plot import DataPlot
from utilities.performance import Performance
from utilities.logger import logger



class SMABacktester(VectorBacktesterBase):

    def __init__(self, filepath, symbol, tc=0.00007, dataset="training", start=None, end=None):
        super().__init__(filepath=filepath, symbol=symbol, start=start, end=end, tc=tc, dataset=dataset)
        self.indicator = "SMA"
        self.perf_obj = Performance(symbol=symbol)
        self.dataploter = DataPlot(dataset, self.indicator, self.symbol)

    def test_strategy(self, freq=5, SMA_S=50, SMA_L=200):  # Adj!!!
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
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L  # NEW!!!

        self.prepare_data(freq, SMA_S, SMA_L)
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
        params_dic = {"freq": freq, "SMA_S": SMA_S, "SMA_L": SMA_L}
        self.dataploter.store_testcase_data(self.perf_obj, params_dic)  # comb[0] is current data freq
        #self.print_performance()

    def prepare_data(self, freq, SMA_S, SMA_L):  # Adj!!!
        ''' Prepares the Data for Backtesting.
        '''
        data = self.data.Close.to_frame().copy()
        freq = "{}min".format(freq)
        resamp = data.resample(freq).last().dropna().iloc[:-1]

        ######### INSERT THE STRATEGY SPECIFIC CODE HERE ##################

        resamp["SMA_S"] = resamp["Close"].rolling(window=SMA_S, min_periods=SMA_S).mean()
        resamp["SMA_L"] = resamp["Close"].rolling(window=SMA_L, min_periods=SMA_L).mean()

        resamp["position"] = np.where(resamp["SMA_S"] > resamp["SMA_L"], 1, 0)

        resamp["position"] = resamp.position.ffill().fillna(0)
        ###################################################################

        resamp.dropna(inplace=True)
        self.results = resamp
        return resamp

    def optimize_strategy(self, freq_range, SMA_S_range, SMA_L_range, metric="Multiple"):  # Adj!!!
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
        sma_s = range(*SMA_S_range)
        sma_l = np.arange(*SMA_L_range)  # NEW!!!

        combinations = list(product(freqs, sma_s, sma_l))
        logger.info(f"sma_backtester: Optimizing of {self.indicator} for {self.symbol} using in total {len(combinations)} combinations..")
        performance = []

        # for data plotting
        #self.strategy_df = pd.DataFrame()

        for comb in combinations:
            if comb[1] < comb[2]:
                self.prepare_data(comb[0], comb[1], comb[2])
                self.upsample()
                self.run_backtest()
                performance.append(performance_function(self.results.strategy))
                # set strategy data and calculate performance
                self.perf_obj.set_data(self.results)
                self.perf_obj.calculate_performance()
                # store strategy performance data for further plotting
                params_dic = {"freq":comb[0], "SMA_S":comb[1], "SMA_L":comb[2]}
                self.dataploter.store_testcase_data(self.perf_obj, params_dic) # comb[0] is current data freq

        #self.results_overview = pd.DataFrame(data=np.array(combinations),
        #                                     columns=["Freq", "EMA_S", "EMA_L"])
        #self.results_overview["Performance"] = performance
        #self.find_best_strategy()

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
    symbol = "XRPUSDT"
    start = ",2021-11-20"
    end = "2022-08-20"
    ptc = 0.00007
    ema = SMABacktester(filepath=filepath, symbol=symbol, start=start, end=end, tc=ptc)
    ema.optimize_strategy((1, 30, 10), (10, 30, 10), (10, 50, 10), metric="Calmar")