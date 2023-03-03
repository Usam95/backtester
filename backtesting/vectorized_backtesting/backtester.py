import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import backtesting.set_project_path

from config.core_config import config
import os

from strategy_functions import execute_ema
from strategy_functions import execute_sma
from strategy_functions import execute_bb
from strategy_functions import execute_macd
from strategy_functions import execute_rsi

import multiprocessing
from multiprocessing import set_start_method
import time

from utilities.logger import Logger
logger = Logger().get_logger()


def check_dirs():
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", config.hist_data_folder))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", config.output_folder))
    if os.path.exists(input_path):
        return
    else:
        logger.error(f"Provided historical data folder {config.hist_data_folder} is not found.")
        exit(0)
    if os.path.exists(output_path):
        return
    else:
        logger.error(f"Provided output folder {config.output_folder} is not found.")
        exit(0)


def print_message(msg, ticker):
    logger.info("*" * 110)
    logger.info(f"Started {msg}-based backtesting for {ticker}.")
    logger.info("*" * 110)


class BackTester:

    def __init__(self):
        check_dirs()
        self.strategies = []
        self.processes = []
        self.execute_strategies()

    def start_process(self, target, ticker):
        process = multiprocessing.Process(target=target, args=(ticker,))
        self.processes.append(process)
        process.start()

    def execute_strategies(self):
        for ticker in config.tickers:
            for strategy in config.strategies:
                if strategy == "ema":
                    self.start_process(execute_ema, ticker)
                    print_message(msg="EMA", ticker=ticker)

                elif strategy == "sma":
                    self.start_process(execute_sma, ticker)
                    print_message(msg="SMA", ticker=ticker)

                elif strategy == "bb":
                    self.start_process(execute_bb, ticker)
                    print_message(msg="BB", ticker=ticker)

                elif strategy == "macd":
                    self.start_process(execute_macd, ticker)
                    print_message(msg="MACD", ticker=ticker)

                elif strategy == "rsi":
                    self.start_process(execute_rsi, ticker)
                    print_message(msg="RSI", ticker=ticker)

        for proc in self.processes:
            proc.join()
        time.sleep(500)


if __name__ == '__main__':

    set_start_method('spawn')
    backtester = BackTester()
