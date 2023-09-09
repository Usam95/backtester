import sys
import os


from matplotlib.backends.backend_pdf import PdfPages


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config.forward_config import create_and_validate_config

import time
from utilities.logger import Logger

logger = Logger().get_logger()

from backtesting.indicators.ema_backtester import EMABacktester
from backtesting.indicators.sma_backtester import SMABacktester
from backtesting.indicators.macd_backtester import MACDBacktester
from backtesting.indicators.rsi_backtester import RsiBacktester
from backtesting.indicators.bb_backtester import BbBacktester
from backtesting.indicators.so_backtester import SoBacktester

from basis_tester import BasisTester


class ForwardTester(BasisTester):

    def __init__(self, config):
        super().__init__(config)

    def perform_test(self, strategy_class, strategy,  symbol, pdf, **strategy_params):
        strategy_name = strategy_class.__name__.replace("Backtester", "").upper()
        self.print_message(msg=strategy_name, ticker=symbol)

        output_dir = self.create_ticker_output_dir(symbol)
        input_file = self.get_hist_data_file(symbol)

        backtester = strategy_class(symbol=symbol, filepath=input_file,
                                    tc=self.config.ptc,
                                    start=self.config.time_frame["start_date"],
                                    end=self.config.time_frame["end_date"])

        backtester.test_strategy(**strategy_params)

        # add the symbol and strategy name to results df
        # Add new columns at the beginning
        backtester.dataploter.df.insert(0, 'strategy', strategy)  # inserting at position 0 (beginning)
        backtester.dataploter.df.insert(0, 'symbol', symbol)
        backtester.dataploter.store_data(output_dir)
        # report performance data of current strategy in the log file
        backtester.perf_obj.print_performance(strategy_name=strategy_name)
        # plot the testing results
        self.plot_results(backtester.perf_obj, pdf, strategy_name, symbol, **strategy_params)

    def execute_tests(self):
        strategy_classes = {
            "macd": MACDBacktester,
            "ema": EMABacktester,
            "sma": SMABacktester,
            "bb": BbBacktester,
            "rsi": RsiBacktester,
            "so": SoBacktester
        }

        # Create a central PDF file
        with PdfPages('Backtest_Results.pdf') as pdf:

            for symbol in self.config.symbols:
                logger.info(f"{symbol=}")
                strategies = self.config.strategies_config.strategies_config
                strategy_config_for_symbol = strategies.get(symbol)

                if not strategy_config_for_symbol:
                    logger.warning(f"No strategy configuration found for symbol: {symbol}")
                    continue

                strategy_config_for_symbol_dict = strategy_config_for_symbol.dict()  # Convert to dictionary

                for strategy_name, params in strategy_config_for_symbol_dict.items():
                    strategy_class = strategy_classes.get(strategy_name)
                    if not strategy_class:
                        logger.warning(f"No strategy class found for strategy: {strategy_name}")
                        continue
                    self.perform_test(strategy_class, strategy_name, symbol, pdf, **params)


if __name__ == '__main__':
    config = create_and_validate_config()
    print(config)
    forward_tester = ForwardTester(config=config)
    forward_tester.execute_tests()
