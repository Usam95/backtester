import sys
import os
import time


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utilities.logger import Logger
logger = Logger().get_logger()
from config.backward_config import create_and_validate_config

from backtesting.vectorized_backtesting.indicators.ema_backtester import EMABacktester
from backtesting.vectorized_backtesting.indicators.sma_backtester import SMABacktester
from backtesting.vectorized_backtesting.indicators.macd_backtester import MACDBacktester
from backtesting.vectorized_backtesting.indicators.rsi_backtester import RsiBacktester
from backtesting.vectorized_backtesting.indicators.bb_backtester import BbBacktester
from backtesting.vectorized_backtesting.indicators.so_backtester import SoBacktester

from basis_tester import BasisTester


class BackwardTester(BasisTester):

    def __init__(self, config):
        super().__init__(config=config)

    def perform_test(self, strategy_class, strategy, symbol, **strategy_params):
        strategy_name = strategy_class.__name__.replace("Backtester", "").upper()
        self.print_message(msg=strategy_name, ticker=symbol)

        start_t = time.time()
        output_dir = self.create_ticker_output_dir(symbol)
        input_file = self.get_hist_data_file(symbol)

        backtester = strategy_class(symbol=symbol, filepath=input_file,
                                    tc=self.config.ptc,
                                    start=self.config.time_frame["start_date"],
                                    end=self.config.time_frame["end_date"])

        backtester.optimize_strategy(**strategy_params,
                                     metric=self.config.metric,
                                     opt_method=self.config.opt_method,
                                     bayesian_trials=self.config.bayesian_trials)

        # add the symbol and strategy name to results df
        # Add new columns at the beginning
        backtester.dataploter.df.insert(0, 'strategy', strategy)  # inserting at position 0 (beginning)
        backtester.dataploter.df.insert(0, 'symbol', symbol)
        backtester.dataploter.store_data(output_dir)

        end_t = time.time()
        self.print_exec_time(start_t, end_t, strategy_class.__name__.replace("Backtester", ""), symbol)

    def execute_tests(self):
        # Mapping of strategy names to their respective classes.
        strategy_classes = {
            "macd": MACDBacktester,
            "ema": EMABacktester,
            "sma": SMABacktester,
            "bb": BbBacktester,
            "rsi": RsiBacktester,
            "so": SoBacktester
        }

        # The frequency is common for all strategies
        freq_range = self.config.strategies_config.freq if hasattr(self.config.strategies_config, 'freq') else []

        strategies = ["ema", "sma", "bb", "macd", "so", "rsi"]
        strategies_config = {strategy: getattr(self.config.strategies_config, strategy) for strategy in strategies if
                             hasattr(self.config.strategies_config, strategy)}

        for symbol in self.config.symbols:
            for strategy_name, strategy_settings in strategies_config.items():
                params = {
                    f"{param}_range": value
                    for param, value in strategy_settings.items()
                }
                # Add the common frequency for all strategies
                params["freq_range"] = freq_range

                strategy_class = strategy_classes.get(strategy_name)
                if strategy_class:
                    self.perform_test(strategy_class, strategy_name, symbol, **params)


if __name__ == '__main__':
    config = create_and_validate_config()
    backward_tester = BackwardTester(config)
    backward_tester.execute_tests()
