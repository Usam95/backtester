import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import backtesting.set_project_path
from config import backward_config
import multiprocessing
from multiprocessing import set_start_method
import time
from utilities.logger import Logger

logger = Logger().get_logger()

from backtesting.indicators.ema_backtester import EMABacktester
from backtesting.indicators.sma_backtester import SMABacktester
from backtesting.indicators.macd_backtester import MACDBacktester
from backtesting.indicators.rsi_backtester import RsiBacktester
from backtesting.indicators.bb_backtester import BbBacktester
from backtesting.indicators.so_backtester import SoBacktester


class BackTester:

    def __init__(self):
        self.check_dirs()
        self.execute_strategies()

    def execute_strategies(self):
        if config.single_test:
            self.perform_single_test()
        else:
            self.perform_optimization()

    def check_dirs(self):
        input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), config.hist_data_folder))
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), config.output_folder))

        # Check if historical data folder exists
        if not os.path.exists(config.hist_data_folder):
            logger.error(f"Provided historical data folder {config.hist_data_folder} is not found. Exiting..")
            exit(1)
        # Check if output folder exists, if not create it
        if not os.path.exists(config.output_folder):
            logger.warning(f"Provided output folder {config.output_folder} is not found. Creating it now.")
            try:
                os.makedirs(output_path)
                config.output_folder = output_path
            except OSError as e:
                logger.error(f"Failed to create the output folder. Reason: {e}.. Exiting..")
                exit(0)

    def create_ticker_output_dir(self, ticker):
        path = os.path.join(config.output_folder, ticker)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_hist_data_file(self, ticker):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), config.hist_data_folder, ticker,
                                            f"{ticker}.parquet.gzip"))
        if not os.path.exists(path):
            logger.error(f"Path {path} for historical data not found.\n"
                         f"Configure data retrieving or provide the correct path.")
            exit(0)
        return path

    def print_exec_time(self, start_t, end_t, tech_ind, ticker):
        total_time = round(((end_t - start_t) / 60), 2)
        logger.info(f"Optimized {tech_ind.upper()} for {ticker} in {total_time} minutes.")

    def print_message(self, msg, ticker):
        logger.info("*" * 110)
        logger.info(f"Started {msg}-based backtesting for {ticker}.")
        logger.info("*" * 110)

    def perform_test(self, strategy_class, strategy,  symbol, single, **strategy_params):
        strategy_name = strategy_class.__name__.replace("Backtester", "").upper()
        self.print_message(msg=strategy_name, ticker=symbol)

        start_t = time.time()
        output_dir = self.create_ticker_output_dir(symbol)
        input_file = self.get_hist_data_file(symbol)

        backtester = strategy_class(symbol=symbol, filepath=input_file,
                                    tc=config.ptc,
                                    start=config.time_frame["start_date"],
                                    end=config.time_frame["end_date"])

        print("In perform test\n")
        print(strategy_params)
        if single:
            backtester.test_strategy(**strategy_params)

        else:
            backtester.optimize_strategy(**strategy_params,
                                         metric=config.metric,
                                         opt_method=config.opt_method,
                                         bayesian_trials=config.bayesian_trials)

        # add the symbol and strategy name to results df
        # Add new columns at the beginning
        backtester.dataploter.df.insert(0, 'strategy', strategy)  # inserting at position 0 (beginning)
        backtester.dataploter.df.insert(0, 'symbol', symbol)
        backtester.dataploter.store_data(output_dir)

        end_t = time.time()
        self.print_exec_time(start_t, end_t, strategy_class.__name__.replace("Backtester", ""), symbol)

    def perform_optimization(self):
        strategies_map = {
            "macd": {
                "class": MACDBacktester,
                "params": {
                    "freq_range": config.strategy_conf.freq,
                    "ema_s_range": config.strategy_conf.macd["ema_s"],
                    "ema_l_range": config.strategy_conf.macd["ema_l"],
                    "signal_mw_range": config.strategy_conf.macd["signal_mw"]
                }
            },

            "ema": {
                "class": EMABacktester,
                "params": {
                    "freq_range": config.strategy_conf.freq,
                    "ema_s_range": config.strategy_conf.ema["ema_s"],
                    "ema_l_range": config.strategy_conf.ema["ema_l"]
                }
            },

            "sma": {
                "class": SMABacktester,
                "params": {
                    "freq_range": config.strategy_conf.freq,
                    "sma_s_range": config.strategy_conf.sma["sma_s"],
                    "sma_l_range": config.strategy_conf.sma["sma_l"]
                }
            },

            "bb": {
                "class": BbBacktester,
                "params": {
                    "freq_range": config.strategy_conf.freq,
                    "window_range": config.strategy_conf.bb["sma"],
                    "dev_range": config.strategy_conf.bb["dev"]
                }
            },

            "rsi": {
                "class": RsiBacktester,
                "params": {
                    "freq_range": config.strategy_conf.freq,
                    "periods_range": config.strategy_conf.rsi["periods"],
                    "rsi_lower_range": config.strategy_conf.rsi["rsi_lower"],
                    "rsi_upper_range": config.strategy_conf.rsi["rsi_upper"]
                }
            },

            "so": {
                "class": SoBacktester,
                "params": {
                    "freq_range": config.strategy_conf.freq,
                    "k_period_range": config.strategy_conf.so["k_period"],
                    "d_period_range": config.strategy_conf.so["d_period"],
                }
            }
        }

        for ticker in config.tickers:
            for strategy in config.strategies:
                details = strategies_map.get(strategy)
                if details:
                    self.perform_test(details["class"], strategy, ticker, single=False, **details["params"])

    def perform_single_test(self):
        strategies_map = {
            "macd": {
                "class": MACDBacktester,
                "params": {
                    "freq": config.single_test_conf.macd["freq"],
                    "ema_s_val": config.single_test_conf.macd["ema_s_val"],
                    "ema_l_val": config.single_test_conf.macd["ema_l_val"],
                    "signal_mw": config.single_test_conf.macd["signal_mw"]
                }
            },

            "ema": {
                "class": EMABacktester,
                "params": {
                    "freq": config.single_test_conf.ema["freq"],
                    "ema_s_val": config.single_test_conf.ema["ema_s_val"],
                    "ema_l_val": config.single_test_conf.ema["ema_l_val"]
                }
            },

            "sma": {
                "class": SMABacktester,
                "params": {
                    "freq": config.single_test_conf.sma["freq"],
                    "sma_s_val": config.single_test_conf.sma["sma_s_val"],
                    "sma_l_val": config.single_test_conf.sma["sma_l_val"]
                }
            },

            "bb": {
                "class": BbBacktester,
                "params": {
                    "freq": config.single_test_conf.bb["freq"],
                    "window": config.single_test_conf.bb["window"],
                    "dev": config.single_test_conf.bb["dev"]
                }
            },

            "rsi": {
                "class": RsiBacktester,
                "params": {
                    "freq": config.single_test_conf.rsi["freq"],
                    "periods": config.single_test_conf.rsi["periods"],
                    "rsi_lower": config.single_test_conf.rsi["rsi_lower"],
                    "rsi_upper": config.single_test_conf.rsi["rsi_upper"]
                }
            },

            "so": {
                "class": SoBacktester,
                "params": {
                    "freq": config.single_test_conf.so["freq"],
                    "k_period": config.single_test_conf.so["k_period"],
                    "d_period": config.single_test_conf.so["d_period"],
                }
            }
        }

        for ticker in config.tickers:
            for strategy in config.strategies:
                details = strategies_map.get(strategy)
                if details:
                    self.perform_test(details["class"], strategy, ticker, single=True, **details["params"])


if __name__ == '__main__':
    set_start_method('spawn')
    backtester = BackTester()
