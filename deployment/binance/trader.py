# Standard library imports
import sys
import os
import time
import threading
from threading import Lock

# Third-party library imports
import numpy as np
import pandas as pd

# Local application/library specific imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '', '..', '..')))
from utilities.logger import Logger
from client_manager import BinanceClientManager
from report_manager import ReportManager
from strategies import Signal, strategy_factory
from config import create_and_validate_config


class SingleAssetTrader:
    def __init__(self, ticker_config, data_completed_event, binance_client_manager, report_manager, logging_lock, logger):

        self.config = ticker_config
        self.client_manager = binance_client_manager
        self.report_manager = report_manager
        self.logger = logger
        self.data_completed_event = data_completed_event
        self.logging_lock = logging_lock
        self.symbol = ticker_config.ticker
        self.bar_length = ticker_config.bar_length
        self.units = ticker_config.units
        self.position = ticker_config.position
        self.stop_loss = ticker_config.stop_loss
        self.take_profit = ticker_config.take_profit

        self.strategies = []
        self.init_strategies()
        # attributes to keep track of trading activity
        self.trades = 0
        self.real_profit = 0
        self.cum_profits = 0
        self.trade_values = []

    def init_strategies(self):
        for strategy_name, strategy_params in self.config.strategies.dict().items():
            if strategy_params:
                self.strategies.append(strategy_factory(strategy_name, **strategy_params))

    def get_trading_signal(self) -> str:
        # Get signals from all strategies
        signals = [(strategy.name, strategy.get_signal(self.client_manager.data)) for strategy in self.strategies]
        buy_strategies = [name for name, signal in signals if signal == Signal.BUY]
        sell_strategies = [name for name, signal in signals if signal == Signal.SELL]

        # Decide on the final signal based on the strategy signals
        # If the number of buy strategies is greater than sell strategies
        if len(buy_strategies) > len(sell_strategies):
            return Signal.BUY, ", ".join(buy_strategies)  # Returns all the names of strategies that suggested BUY

        # If the number of sell strategies is greater than buy strategies
        elif len(sell_strategies) > len(buy_strategies):
            return Signal.SELL, ", ".join(sell_strategies)  # Returns all the names of strategies that suggested SELL

        # If the number of both signals is identical
        return Signal.NEUTRAL, ""

    def update_trading_data(self, trading_data):
        # extract data from order object
        order = trading_data['order']
        side = order["side"]
        time = pd.to_datetime(order["transactTime"], unit="ms")
        base_units = float(order["executedQty"])
        quote_units = float(order["cummulativeQuoteQty"])
        price = round(quote_units / base_units, 5)

        self.trades += 1

        if side == "BUY":
            self.trade_values.append(-quote_units)
        elif side == "SELL":
            self.trade_values.append(quote_units)

        if self.trades % 2 == 0:#
            # represents the profit from the last two trades.
            self.real_profit = round(np.sum(self.trade_values[-2:]), 3)
            # represents the cumulative profit from all the trades since client start
            self.cum_profits = round(np.sum(self.trade_values), 3)
        else:
            self.real_profit = 0
            # compute the cumulative profit of all trades except the last one.
            self.cum_profits = round(np.sum(self.trade_values[:-1]), 3)

        # Update the trading_data dictionary with new values
        trading_data.update({
            "side": side,
            "time": time,
            "base_units": base_units,
            "quote_units": quote_units,
            "price": price,
            "total_trades": self.trades,
            "real_profit": self.real_profit,
            "cumulative_profits": self.cum_profits,
        })

        return trading_data

    def execute_trades(self):

        current_signal, strategy_names = self.get_trading_signal()
        self.logger.debug(f"Executing trade for symbol {self.symbol} using strategies: {strategy_names} at {time.ctime()}")
        trading_data = {
            'strategy_names': strategy_names,
            'ticker': self.symbol,
        }

        if current_signal == Signal.BUY and self.position == 0:
            order = self.client_manager.client.create_order(symbol=self.symbol, side="BUY", type="MARKET",
                                                            quantity=self.units)

            if order['status'] == 'FILLED':
                trade_action = "GOING LONG"
                trading_data.update({
                    'order': order,
                    'trade_action': trade_action,
                })
                self.position = 1

                # After successfully buying, place a TAKE_PROFIT and STOP_LOSS order
                executed_price = float(order['fills'][0]['price'])
                stop_loss_price = executed_price * (1 - self.stop_loss)
                take_profit_price = executed_price * (1 + self.take_profit)

                # Place STOP_LOSS order and store its ID
                stop_loss_order = self.client_manager.client.create_order(symbol=self.symbol, side="SELL",
                                                                          type="STOP_LOSS",
                                                                          quantity=self.units,
                                                                          stopPrice=stop_loss_price)
                self.stop_loss_order_id = stop_loss_order['orderId']

                # Place TAKE_PROFIT order and store its ID
                take_profit_order = self.client_manager.client.create_order(symbol=self.symbol, side="SELL",
                                                                            type="TAKE_PROFIT",
                                                                            quantity=self.units,
                                                                            stopPrice=take_profit_price)
                self.take_profit_order_id = take_profit_order['orderId']

                trading_data = self.update_trading_data(trading_data)
                # report the trade
                with self.logging_lock:
                    self.report_manager.log_trade_data(trading_data)

        elif current_signal == Signal.SELL and self.position == 1:
            order = self.client_manager.client.create_order(symbol=self.symbol, side="SELL", type="MARKET",
                                                            quantity=self.units)

            if order['status'] == 'FILLED':
                trade_action = "GOING NEUTRAL"
                trading_data.update({
                    'order': order,
                    'trade_action': trade_action,
                })
                self.position = 0

                # After selling, cancel any pending TAKE_PROFIT or STOP_LOSS orders
                if hasattr(self, 'stop_loss_order_id'):  # Check if attribute exists
                    self.client_manager.client.cancel_order(symbol=self.symbol, orderId=self.stop_loss_order_id)
                    delattr(self, 'stop_loss_order_id')

                if hasattr(self, 'take_profit_order_id'):  # Check if attribute exists
                    self.client_manager.client.cancel_order(symbol=self.symbol, orderId=self.take_profit_order_id)
                    delattr(self, 'take_profit_order_id')

                trading_data = self.update_trading_data(trading_data)
                # report the trade
                with self.logging_lock:
                    self.report_manager.log_trade_data(trading_data)


class LongOnlyTrader:

    def __init__(self):  # your initialization parameters

        self.config = create_and_validate_config()
        self.tickers_config = self.config.tickers_config
        self.data_completed_events = {ticker_config.ticker: threading.Event() for ticker_config in self.tickers_config}
        self.client_manager = BinanceClientManager(self.data_completed_events, testnet=True)
        self.report_manager = ReportManager()
        self.logger = Logger()
        self.logging_lock = Lock()

        self.running = True  # A flag to control threads
        self.trader_threads = []  # Store threads to join later
        # Initialize the traders list
        self.traders = []
        self.init_traders()

        # start binance trader:
        self.client_manager.connect_client()
        self.client_manager.start_trading(self.config.tickers_config)

    def init_traders(self):
        for ticker_config in self.tickers_config:
            trader = SingleAssetTrader(ticker_config, self.data_completed_events[ticker_config.ticker],
                                       self.client_manager, self.report_manager, self.logging_lock, self.logger)
            self.traders.append(trader)

    def trade_ticker(self, trader):
        while self.running:
            trader.data_completed_event.wait()  # Wait for individual ticker data completion
            trader.execute_trades()
            trader.data_completed_event.clear()  # Clear the event

    def start_trading(self):
        # Spawn a new thread for each trader
        for trader in self.traders:
            ticker_thread = threading.Thread(target=self.trade_ticker, args=(trader,))
            ticker_thread.start()
            self.trader_threads.append(ticker_thread)

    def stop_trading(self):
        # Signal all threads to stop
        self.running = False

        # Set all events to ensure threads wake up from their wait state
        for event in self.data_completed_events.values():
            event.set()

        # Now join the threads
        for thread in self.trader_threads:
            thread.join()


if __name__ == "__main__":
    trader_instance = LongOnlyTrader()
    try:
        trader_instance.start_trading()
        # Optionally, you can use a sleep here for demo purposes
        # time.sleep(60)  # run for 60 seconds
    except KeyboardInterrupt:
        print("Stopping traders...")
        trader_instance.stop_trading()