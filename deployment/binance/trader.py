# Standard library imports
import sys
import os
import time
import threading
from threading import Lock
import math
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
    def __init__(self, id, ticker_config, data_completed_event, binance_client_manager, report_manager, logging_lock, logger):
        self.id = id
        self.config = ticker_config
        self.client_manager = binance_client_manager
        self.report_manager = report_manager
        self.logging = logger
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
                self.logging.info(f"Trader ID {self.id}: got the strategy: {strategy_name}")
                self.strategies.append(strategy_factory(strategy_name, **strategy_params))

    def get_trading_signal(self) -> str:
        # Get signals from all strategies
        signals = [(strategy.strategy_name, strategy.get_signal(self.client_manager.data[self.symbol][self.bar_length])) for strategy in self.strategies]
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

    def adjust_precision(self, symbol, price=None, quantity=None):
        # Fetch trading rules for the symbol
        # This is a simplified example, you may need to fetch this once and cache it
        # rather than fetching every time for efficiency.
        try:
            exchange_info = self.client_manager.client.get_exchange_info()
            symbol_info = next((s for s in exchange_info["symbols"] if s["symbol"] == symbol), None)

            if not symbol_info:
                raise ValueError(f"No trading rules found for symbol {symbol}")

            # Get the maximum number of decimal places allowed for price and quantity
            price_filter = next((f for f in symbol_info["filters"] if f["filterType"] == "PRICE_FILTER"), None)
            lot_size_filter = next((f for f in symbol_info["filters"] if f["filterType"] == "LOT_SIZE"), None)

            if price_filter:
                price_precision = len(price_filter["tickSize"].rstrip('0').split('.')[1])
            if lot_size_filter:
                quantity_precision = len(lot_size_filter["stepSize"].rstrip('0').split('.')[1])

            if price:
                price = round(price, price_precision)

            if quantity:
                quantity = round(quantity, quantity_precision)

            return price, quantity

        except Exception as e:
            self.logging.error(f"Error adjusting precision for symbol {symbol}: {e}")
            raise

    def is_order_type_supported(self, symbol, order_type):
        try:
            exchange_info = self.client_manager.client.get_exchange_info()
            symbol_info = next((s for s in exchange_info["symbols"] if s["symbol"] == symbol), None)

            if not symbol_info:
                raise ValueError(f"No trading rules found for symbol {symbol}")

            return order_type in symbol_info['orderTypes']

        except Exception as e:
            self.logging.error(f"Error checking order type support for symbol {symbol}: {e}")
            raise

    def place_stop_loss_order(self, order):
        try:
            # Check if STOP_LOSS order type is supported
            stop_loss_supported = self.is_order_type_supported(self.symbol, 'STOP_LOSS')
            self.logging.info(f"STOP_LOSS supported for {self.symbol}: {stop_loss_supported}")
            executed_price = float(order['fills'][0]['price'])
            stop_loss_price = executed_price * (1 - self.stop_loss)
            # adjust the prices and units according to binance requirements
            adj_stop_loss_price, _ = self.adjust_precision(self.symbol, price=stop_loss_price)
            _, adj_units = self.adjust_precision(self.symbol, quantity=self.units)

            self.logging.info(f"Adjusted STOP_LOSS price for {self.symbol}: {adj_stop_loss_price}")
            self.logging.info(f"Adjusted units for {self.symbol}: {adj_units}")

            if stop_loss_supported:
                # Place STOP_LOSS order and store its ID
                stop_loss_order = self.client_manager.client.create_order(symbol=self.symbol, side="SELL",
                                                                          type="STOP_LOSS",
                                                                          quantity=adj_units,
                                                                          stopPrice=adj_stop_loss_price)
                # If STOP_LOSS is not supported, place custom STOP_LOSS LIMIT order and store its ID
            else:
                stop_loss_order = self.client_manager.client.create_order(symbol=self.symbol, side="SELL",
                                                                              type="LIMIT", timeInForce="GTC",
                                                                              quantity=adj_units,
                                                                              price=adj_stop_loss_price)
            self.stop_loss_order_id = stop_loss_order['orderId']

        except Exception as e:
            self.logging.error(f"Error placing stop loss order: {e}")
            raise

    def place_take_profit_order(self, order):
        try:
            # Check if TAKE_PROFIT order type is supported
            take_profit_supported = self.is_order_type_supported(self.symbol, 'TAKE_PROFIT')
            self.logging.info(f"TAKE_PROFIT supported for {self.symbol}: {take_profit_supported}")

            executed_price = float(order['fills'][0]['price'])
            take_profit_price = executed_price * (1 + self.take_profit)

            # adjust the prices and units according to binance requirements
            adj_take_profit_price, _ = self.adjust_precision(self.symbol, price=take_profit_price)
            _, adj_units = self.adjust_precision(self.symbol, quantity=self.units)

            self.logging.info(f"Adjusted TAKE_PROFIT price for {self.symbol}: {adj_take_profit_price}")
            self.logging.info(f"Adjusted units for {self.symbol}: {adj_units}")

            if take_profit_supported:
                # Place TAKE_PROFIT order and store its ID
                take_profit_order = self.client_manager.client.create_order(symbol=self.symbol, side="SELL",
                                                                            type="TAKE_PROFIT",
                                                                            quantity=adj_units,
                                                                            stopPrice=adj_take_profit_price)
            else:
                # Place TAKE_PROFIT order and store its ID
                take_profit_order = self.client_manager.client.create_order(symbol=self.symbol, side="SELL",
                                                                              type="LIMIT", timeInForce="GTC",
                                                                              quantity=adj_units,
                                                                              price=adj_take_profit_price)
            self.take_profit_order_id = take_profit_order['orderId']

        except Exception as e:
            self.logging.error(f"Error placing take profit order: {e}")
            raise

    def execute_trades(self):
        try:
            current_signal, strategy_names = self.get_trading_signal()
            self.logging.info(f"Trader ID: {self.id}: Executing trade for symbol {self.symbol} using strategies: {strategy_names} at {time.ctime()}")
            trading_data = {
                'strategy_names': strategy_names,
                'ticker': self.symbol
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

                    # update the trading data
                    trading_data = self.update_trading_data(trading_data)
                    # report the trade
                    with self.logging_lock:
                        self.report_manager.log_trade_data(trading_data)

                    # After successfully buying, place a TAKE_PROFIT and STOP_LOSS order
                    self.place_stop_loss_order(order)
                    self.place_take_profit_order(order)

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

        except Exception as e:
            self.logging.error(f"Error executing trades: {e}")
            raise


class LongOnlyTrader:

    def __init__(self):  # your initialization parameters

        self.config = create_and_validate_config()
        self.tickers_config = self.config.tickers_config
        self.data_completed_events = {ticker_config.ticker: threading.Event() for ticker_config in self.tickers_config}
        self.client_manager = BinanceClientManager(self.data_completed_events, testnet=True)
        self.report_manager = ReportManager()
        self.logging = Logger().get_logger()
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
        for id, ticker_config in enumerate(self.tickers_config, start=1):
            trader = SingleAssetTrader(id, ticker_config, self.data_completed_events[ticker_config.ticker],
                                       self.client_manager, self.report_manager, self.logging_lock, self.logging)
            self.traders.append(trader)

    def trade_ticker(self, trader):
        while self.running:
            trader.data_completed_event.wait()  # Wait for individual ticker data completion
            self.logging.debug(f"Trader for {trader.symbol} woked by data_completed_event{trader.data_completed_event}..")
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