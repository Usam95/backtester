from core_config import config
import os
from strategy_functions import execute_ema
from strategy_functions import execute_sma
from strategy_functions import execute_bb
from strategy_functions import execute_macd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import multiprocessing
from multiprocessing import set_start_method

import time

class Backtester:

    def __init__(self):

        self.check_dirs()
        self.strategies = []
        self.processes = []
        self.execute_strategies()


    def check_dirs(self):
        if os.path.exists(config.hist_data_folder):
            return
        else:
            print(f"ERROR: provided historical data folder {config.hist_data_folder} is not found")
            exit(0)
        if os.path.exists(config.output_folder):
            return
        else:
            print(f"ERROR: provided output folder {config.output_folder} is not found")
            exit(0)

    def start_process(self, target, ticker):
        process = multiprocessing.Process(target=target, args=(ticker,))
        self.processes.append(process)
        process.start()

    def execute_strategies(self):
        for ticker in config.tickers:
            for strategy in config.strategies:
                if strategy == "ema":
                    self.start_process(execute_ema, ticker)
                    print(f"Started ema backtesting..", flush=True)

                elif strategy == "sma":
                    self.start_process(execute_sma, ticker)
                    print(f"Started sma backtesting..", flush=True)

                elif strategy == "bb":
                    self.start_process(execute_bb, ticker)
                    print(f"Started bb backtesting..", flush=True)
                elif strategy == "macd":
                    self.start_process(execute_macd, ticker)
                    print(f"Started macd backtesting..", flush=True)

        for proc in self.processes:
            proc.join()
        time.sleep(500)


if __name__ == '__main__':
    set_start_method('fork')
    backtester = Backtester()
