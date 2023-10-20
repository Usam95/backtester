import os
import time

from backtesting.vectorized_backtesting.indicators.ema_backtester import EMABacktester
from backtesting.vectorized_backtesting.indicators.sma_backtester import SMABacktester
from backtesting.vectorized_backtesting.indicators.macd_backtester import MACDBacktester
from backtesting.vectorized_backtesting.indicators.rsi_backtester import RSIBacktester
from backtesting.vectorized_backtesting.indicators.bb_backtester import BBBacktester
from sys import exit

from utilities.logger import Logger
logger = Logger().get_logger()


def create_ticker_output_dir(ticker):
    #path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", config.hist_data_folder))
    path = os.path.join(config.output_folder, ticker)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_hist_data_file(ticker):
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", config.hist_data_folder, ticker, f"{ticker}.parquet.gzip"))
    # path = os.path.join(os.path.realpath(config.hist_data_folder), ticker, f"{ticker}.parquet.gzip")
    if not os.path.exists(path):
        logger.error(f"Path {path} for historical data not found.\n"
                     f"Configure data retrieving or provide the correct path.")
        exit(0)
    return path


def print_exec_time(start_t, end_t, tech_ind, ticker):
    total_time = {round(((end_t - start_t) / 60), 2)}
    logger.info(f"Optimized {tech_ind.upper()} for {ticker} in {total_time} minutes.")


def execute_macd(ticker):
    start_t = time.time()
    output_dir = create_ticker_output_dir(ticker)
    input_file = get_hist_data_file(ticker)

    macd_backtester = MACDBacktester(symbol=ticker, filepath=input_file,
                                     tc=config.ptc,
                                     start=config.time_frame["start_date"],
                                     end=config.time_frame["end_date"])

    macd_backtester.optimize_strategy(freq_range=config.strategy_conf.freq,
                                      ema_s_range=config.strategy_conf.macd["ema_s"],
                                      ema_l_range=config.strategy_conf.macd["ema_l"],
                                      signal_mw_range=config.strategy_conf.macd["signal_mw"],
                                      metric=config.strategy_conf.metric)

    macd_backtester.dataploter.store_data(output_dir)
    end_t = time.time()
    print_exec_time(start_t, end_t, "macd", ticker)


def execute_ema(ticker):
    start_t = time.time()
    output_dir = create_ticker_output_dir(ticker)
    input_file = get_hist_data_file(ticker)

    ema_backtester = EMABacktester(symbol=ticker, filepath=input_file,
                                   tc=config.ptc,
                                   start=config.time_frame["start_date"],
                                   end=config.time_frame["end_date"])

    ema_backtester.optimize_strategy(freq_range=config.strategy_conf.freq,
                                     ema_s_range=config.strategy_conf.ema["ema_s"],
                                     ema_l_range=config.strategy_conf.ema["ema_l"],
                                     metric=config.strategy_conf.metric)

    ema_backtester.dataploter.store_data(output_dir)
    end_t = time.time()
    print_exec_time(start_t, end_t, "ema", ticker)


def execute_sma(ticker):
    start_t = time.time()
    output_dir = create_ticker_output_dir(ticker)
    input_file = get_hist_data_file(ticker)

    sma_backtester = SMABacktester(symbol=ticker, filepath=input_file,
                                   tc=config.ptc,
                                   start=config.time_frame["start_date"],
                                   end=config.time_frame["end_date"])

    sma_backtester.optimize_strategy(freq_range=config.strategy_conf.freq,
                                     sma_s_range=config.strategy_conf.sma["sma_s"],
                                     sma_l_range=config.strategy_conf.sma["sma_l"],
                                     metric=config.strategy_conf.metric)

    sma_backtester.dataploter.store_data(output_dir)
    end_t = time.time()
    print_exec_time(start_t, end_t, "sma", ticker)


def execute_bb(ticker):
    start_t = time.time()
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
    end_t = time.time()
    print_exec_time(start_t, end_t, "bb", ticker)


def execute_rsi(ticker):
    start_t = time.time()
    output_dir = create_ticker_output_dir(ticker)
    input_file = get_hist_data_file(ticker)

    rsi_backtester = RSIBacktester(symbol=ticker, filepath=input_file,
                                   tc=config.ptc,
                                   start=config.time_frame["start_date"],
                                   end=config.time_frame["end_date"])

    rsi_backtester.optimize_strategy(freq_range=config.strategy_conf.freq,
                                     periods_range=config.strategy_conf.rsi["periods"],
                                     rsi_lower_range=config.strategy_conf.rsi["rsi_lower"],
                                     rsi_upper_range=config.strategy_conf.rsi["rsi_upper"],
                                     metric=config.strategy_conf.metric)

    rsi_backtester.dataploter.store_data(output_dir)
    end_t = time.time()
    print_exec_time(start_t, end_t, "rsi", ticker)
