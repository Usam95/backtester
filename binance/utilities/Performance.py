import pandas as pd
import numpy as np


class Performance:
    def __init__(self, data=None, symbol=None):
        if data is None and symbol is None:
            print("WARNING: Data and symbol were not provided to constructor\n"
                  "Please use the function 'set_data'. to set data to be analysed.")
        else:
            self.symbol = symbol
            self.data = data
            self.to_analyze = self.data.strategy
            self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))

    def set_data(self, data, symbol):
        self.data = data.copy()
        self.symbol = symbol
        self.to_analyze = self.data.strategy

        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))

    def calculate_performance(self):

        self.strategy_multiple = round(self.calculate_multiple(self.to_analyze), 6)
        self.bh_multiple = round(self.calculate_multiple(self.data.returns), 6)
        self.outperf = round(self.strategy_multiple - self.bh_multiple, 6)
        self.cagr = round(self.calculate_cagr(self.to_analyze), 6)
        self.ann_mean = round(self.calculate_annualized_mean(self.to_analyze), 6)
        self.ann_std = round(self.calculate_annualized_std(self.to_analyze), 6)
        self.sharpe = round(self.calculate_sharpe(self.to_analyze), 6)
        self.sortino = round(self.calculate_sortino(self.to_analyze), 6)
        self.max_drawdown = round(self.calculate_max_drawdown(self.to_analyze), 6)
        self.calmar = round(self.calculate_calmar(self.to_analyze), 6)
        self.max_dd_duration = round(self.calculate_max_dd_duration(self.to_analyze), 6)
        self.kelly_criterion = round(self.calculate_kelly_criterion(self.to_analyze), 6)
        self.num_of_trades = self.calclulate_num_of_trades()

    def print_performance(self):
        ''' Calculates and prints various Performance Metrics.
        '''
        print(100 * "=")
        # print("SIMPLE CONTRARIAN STRATEGY | INSTRUMENT = {} | Freq: {} | WINDOW = {}".format(self.symbol, self.freq, self.window))
        print("SIMPLE CONTRARIAN STRATEGY | INSTRUMENT = {}".format(self.symbol))

        print(100 * "-")
        # print("\n")
        print("PERFORMANCE MEASURES:")
        print("\n")
        print(f"Number of trades: {self.num_of_trades}")
        print("Multiple (Strategy):         {}".format(self.strategy_multiple))
        print("Multiple (Buy-and-Hold):     {}".format(self.bh_multiple))
        print(38 * "-")
        print("Out-/Underperformance:       {}".format(self.outperf))
        print("\n")
        print("CAGR:                        {}".format(self.cagr))
        print("Annualized Mean:             {}".format(self.ann_mean))
        print("Annualized Std:              {}".format(self.ann_std))
        print("Sharpe Ratio:                {}".format(self.sharpe))
        print("Sortino Ratio:               {}".format(self.sortino))
        print("Maximum Drawdown:            {}".format(self.max_drawdown))
        print("Calmar Ratio:                {}".format(self.calmar))
        print("Max Drawdown Duration:       {} Days".format(self.max_dd_duration))
        print("Kelly Criterion:             {}".format(self.kelly_criterion))

        print(100 * "=")

    ############################## Performance ######################################
    def calclulate_num_of_trades(self):
        return self.data.trades.value_counts().to_dict()[1.0]

    def calculate_multiple(self, series):
        return np.exp(series.sum())

    def calculate_cagr(self, series):
        return np.exp(series.sum()) ** (1 / ((series.index[-1] - series.index[0]).days / 365.25)) - 1

    def calculate_annualized_mean(self, series):
        return series.mean() * self.tp_year

    def calculate_annualized_std(self, series):
        return series.std() * np.sqrt(self.tp_year)

    def calculate_sharpe(self, series):
        if series.std() == 0:
            return np.nan
        else:
            return series.mean() / series.std() * np.sqrt(self.tp_year)

    def calculate_sortino(self, series):
        excess_returns = (series - 0)
        downside_deviation = np.sqrt(np.mean(np.where(excess_returns < 0, excess_returns, 0) ** 2))
        if downside_deviation == 0:
            return np.nan
        else:
            sortino = (series.mean() - 0) / downside_deviation * np.sqrt(self.tp_year)
            return sortino

    def calculate_max_drawdown(self, series):
        creturns = series.cumsum().apply(np.exp)
        cummax = creturns.cummax()
        drawdown = (cummax - creturns) / cummax
        max_dd = drawdown.max()
        return max_dd

    def calculate_calmar(self, series):
        max_dd = self.calculate_max_drawdown(series)
        if max_dd == 0:
            return np.nan
        else:
            cagr = self.calculate_cagr(series)
            calmar = cagr / max_dd
            return calmar

    def calculate_max_dd_duration(self, series):
        creturns = series.cumsum().apply(np.exp)
        cummax = creturns.cummax()
        drawdown = (cummax - creturns) / cummax

        begin = drawdown[drawdown == 0].index
        end = begin[1:]
        end = end.append(pd.DatetimeIndex([drawdown.index[-1]]))
        periods = end - begin
        max_ddd = periods.max()
        return max_ddd.days

    def calculate_kelly_criterion(self, series):
        series = np.exp(series) - 1
        if series.var() == 0:
            return np.nan
        else:
            return series.mean() / series.var()

    ############################## Performance ######################################