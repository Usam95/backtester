import pandas as pd
import numpy as np

from utilities.logger import Logger
logger = Logger().get_logger()


class Performance:
    def __init__(self, data=None, symbol=None,  multiple_only = True):

        self.symbol = symbol
        self.multiple_only = multiple_only

        self.strategy_multiple_net = None
        self.strategy_multiple = None
        self.bh_multiple = None
        self.outperf_net = None
        self.outperf = None
        self.num_of_trades = None
        self.num_samples = None

        self.cagr = None
        self.ann_mean = None
        self.ann_std = None
        self.sharpe = None
        self.sortino = None
        self.max_drawdown = None
        self.calmar = None
        self.max_dd_duration = None
        self.kelly_criterion  = None

        if data is None:
            print("WARNING: Data and symbol were not provided to constructor\n"
                  "Please use the function 'set_data'. to set data to be analysed.")
        else:
            self.data = data
            self.to_analyze = self.data.strategy
            self.returns = self.data.returns
            self.tp_year = 1    # default value for number of years

        self.performance_functions = {
            "Multiple": self.calculate_multiple,
            "Sharpe": self.calculate_sharpe,
            "Sortino": self.calculate_sortino,
            "Calmar": self.calculate_calmar,
            "Kelly": self.calculate_kelly_criterion
        }

    def calculate_tp_year(self):
        timeframe_in_days = (self.data.index[-1] - self.data.index[0]).days
        if timeframe_in_days > 0:
            self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))

    def set_strategy_data(self, strategy, returns):
        self.to_analyze = strategy
        self.returns = returns

    def set_data(self, data):
        self.data = data.copy()
        self.returns = self.data.returns
        self.tp_year = (self.data.Close.count() / ((self.data.index[-1] - self.data.index[0]).days / 365.25))

    def calculate_performance(self):

        self.strategy_multiple_net = round(self.calculate_multiple(self.data.strategy_net), 3)
        self.strategy_multiple = round(self.calculate_multiple(self.data.strategy), 3)
        self.bh_multiple = round(self.calculate_multiple(self.returns), 3)
        self.outperf_net = round(self.strategy_multiple_net - self.bh_multiple, 3)
        self.outperf = round(self.strategy_multiple - self.bh_multiple, 3)
        self.num_of_trades = self.calculate_num_of_trades()
        self.num_samples = len(self.data)
        if not self.multiple_only:
            self.cagr = round(self.calculate_cagr(self.strategy_net), 3)
            self.ann_mean = round(self.calculate_annualized_mean(self.data.strategy_net), 3)
            self.ann_std = round(self.calculate_annualized_std(self.data.strategy_net), 3)
            self.sharpe = round(self.calculate_sharpe(self.data.strategy_net), 3)
            self.sortino = round(self.calculate_sortino(self.data.strategy_net), 3)
            self.max_drawdown = round(self.calculate_max_drawdown(self.data.strategy_net), 3)
            self.calmar = round(self.calculate_calmar(self.data.strategy_net), 3)
            self.max_dd_duration = round(self.calculate_max_dd_duration(self.data.strategy_net), 3)
            self.kelly_criterion = round(self.calculate_kelly_criterion(self.data.strategy_net), 3)
            # store some attributes of historical data
            self.init_histdata_attr()

    def init_histdata_attr(self):
        self.start_d = self.data.index[0]
        self.end_d = self.data.index[-1]

    def print_performance(self, strategy_name):
        self.calculate_performance()
        ''' Calculates and prints various Performance Metrics.
        '''

        logger.info(100 * "=")
        logger.info(f"{strategy_name.upper()} STRATEGY | INSTRUMENT = {self.symbol}")
        logger.info(100 * "-")
        logger.info("PERFORMANCE MEASURES:")

        # Note: Adjust the width value (e.g., 30) if needed
        logger.info(f"{'Multiple (Strategy):':<30} {self.strategy_multiple:>10.3f}")
        logger.info(f"{'Multiple (Buy-and-Hold):':<30} {self.bh_multiple:>10.3f}")
        logger.info(f"{'Number of Trades:':<30} {self.num_of_trades:>10}")
        logger.info(f"{'Strategy Multiple Net:':<30} {self.strategy_multiple_net:>10.3f}")
        logger.info(f"{'Net Out-/Under performance:':<30} {self.outperf_net:>10.3f}")

        logger.info(100 * "=")
        logger.info(100 * "=")

        if not self.multiple_only:
            logger.info("\n")
            logger.info(f"{'CAGR:':<30} {self.cagr:>10.3f}")
            logger.info(f"{'Annualized Mean:':<30} {self.ann_mean:>10.3f}")
            logger.info(f"{'Annualized Std:':<30} {self.ann_std:>10.3f}")
            logger.info(f"{'Sharpe Ratio:':<30} {self.sharpe:>10.3f}")
            logger.info(f"{'Sortino Ratio:':<30} {self.sortino:>10.3f}")
            logger.info(f"{'Maximum Drawdown:':<30} {self.max_drawdown:>10.3f}")
            logger.info(f"{'Calmar Ratio:':<30} {self.calmar:>10.3f}")
            logger.info(f"{'Max Drawdown Duration:':<30} {self.max_dd_duration:>10} Days")
            logger.info(f"{'Kelly Criterion:':<30} {self.kelly_criterion:>10.3f}")
            logger.info(100 * "=")

    # ========================== Performance ==========================

    def calculate_num_of_trades(self):
        return int(self.data.trades.value_counts().get(1.0, 0))

    def calculate_multiple(self, series):
        return np.exp(series.sum())

    def calculate_cagr(self, series):
        if (series.index[-1] - series.index[0]).days > 0:
            return np.exp(series.sum()) ** (1 / ((series.index[-1] - series.index[0]).days / 365.25)) - 1
        else:
            return 1
        
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
