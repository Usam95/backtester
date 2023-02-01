from core_config import config
import os
from strategies.ema_backtester import EMABacktester
from strategies.sma_backtester import SMABacktester
from strategies.macd_backtester import MACDBacktester

from strategies.bb_blacktester import BBBacktester
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ''))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))




def create_ticker_output_dir(ticker):
    path = os.path.join(config.output_folder, ticker)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_hist_data_file(ticker):
    path = os.path.join(config.hist_data_folder, ticker, f"{ticker}.parquet.gzip")
    if not os.path.exists(path):
        # and not config.retrieve_data:
        print(f"ERROR: Historical data are not found. Please configure retrieving data in config file.")
        print(f"{path=}")
        exit(0)
    return path

def execute_macd(ticker):
    output_dir = create_ticker_output_dir(ticker)
    input_file = get_hist_data_file(ticker)

    macd_backtester = MACDBacktester(symbol=ticker, filepath=input_file,
                                   tc=config.ptc,
                                   start=config.time_frame["start_date"],
                                   end=config.time_frame["end_date"])

    macd_backtester.optimize_strategy(freq_range=config.strategy_conf.freq,
                                     EMA_S_range=config.strategy_conf.macd["ema_s"],
                                     EMA_L_range=config.strategy_conf.macd["ema_l"],
                                     signal_mw_range=config.strategy_conf.macd["signal_mw"],
                                     metric=config.strategy_conf.metric)

    macd_backtester.dataploter.store_data(output_dir)
def execute_ema(ticker):
    output_dir = create_ticker_output_dir(ticker)
    input_file = get_hist_data_file(ticker)

    ema_backtester = EMABacktester(symbol=ticker, filepath=input_file,
                                   tc=config.ptc,
                                   start=config.time_frame["start_date"],
                                   end=config.time_frame["end_date"])

    ema_backtester.optimize_strategy(freq_range=config.strategy_conf.freq,
                                     EMA_S_range=config.strategy_conf.ema["ema_s"],
                                     EMA_L_range=config.strategy_conf.ema["ema_l"],
                                     metric=config.strategy_conf.metric)

    ema_backtester.dataploter.store_data(output_dir)


def execute_sma(ticker):
    output_dir = create_ticker_output_dir(ticker)
    input_file = get_hist_data_file(ticker)

    sma_backtester = SMABacktester(symbol=ticker, filepath=input_file,
                                   tc=config.ptc,
                                   start=config.time_frame["start_date"],
                                   end=config.time_frame["end_date"])

    sma_backtester.optimize_strategy(freq_range=config.strategy_conf.freq,
                                     SMA_S_range=config.strategy_conf.sma["sma_s"],
                                     SMA_L_range=config.strategy_conf.sma["sma_l"],
                                     metric=config.strategy_conf.metric)

    sma_backtester.dataploter.store_data(output_dir)


def execute_bb(ticker):
    output_dir = create_ticker_output_dir(ticker)
    input_file = get_hist_data_file(ticker)

    bb_backtester = BBBacktester(symbol=ticker, filepath=input_file,
                                 tc=config.ptc,
                                 start=config.time_frame["start_date"],
                                 end=config.time_frame["end_date"])

    bb_backtester.optimize_strategy(freq_range=config.strategy_conf.freq,
                                    window_range=config.strategy_conf.bb["sma"],
                                    dev_range=config.strategy_conf.bb["dev"],
                                    metric=config.strategy_conf.metric)

    bb_backtester.dataploter.store_data(output_dir)
