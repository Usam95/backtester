# Standard library imports
import sys
import os
from datetime import datetime, timedelta
import time

# Third-party library imports
import pandas as pd
from binance.client import Client
from binance import ThreadedWebsocketManager

# Update sys.path for local application imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '', '..', '..')))

# Local application/library specific imports
from utilities.credentials import api_key, secret_key, testnet_api_key, testnet_secret_key
from utilities.logger import Logger


MAX_ATTEMPTS = 35
WAIT_TIME = 15
WAIT_INCREASE = 15
MAX_WAIT = 300


class BinanceClientManager:
    def __init__(self, data_completed_events, testnet=True):
        self.client = None
        self.twm = ThreadedWebsocketManager()
        self.data_completed_events = data_completed_events
        self.testnet = testnet
        self.wait = WAIT_TIME
        self.wait_increase = WAIT_INCREASE
        self.data = {}
        self.logger = Logger().get_logger()
        self.logger.debug("Binance client was initialised successfully.")

        #self.connect_client()
        #self.start_trading()

    def connect_client(self):
        """Establish a connection with the Binance client."""
        credentials = (testnet_api_key, testnet_secret_key) if self.testnet else (api_key, secret_key)
        try:
            self.client = Client(api_key=credentials[0], api_secret=credentials[1], tld="com", testnet=self.testnet)
            self.twm.start()
            self.logger.debug("Binance client was started successfully.")
        except Exception as e:
            self.logger.error(f"Error occurred during connection establishment with server: {e}")
            self.logger.error(f"Exiting...")
            sys.exit(1)

    def client_is_connected(self, reconnect=True, verbose=True) -> bool:
        """Check if the client is connected to Binance."""
        try:
            if len(self.client.ping()) == 0:
                if verbose:
                    self.logger.info("Connection is ok.")
                return True
        except Exception as e:
            if verbose:
                self.logger.error(f"Connection with Binance server was lost due to: {e}. Trying to reconnect..")
            if reconnect:
                return self.reconnect_client()
            else:
                return False

    def stop_all(self):
        """Stop the ThreadedWebsocketManager and any other running threads."""
        self.twm.stop()

    def reconnect_client(self) -> bool:
        """Attempt to reconnect to Binance if the connection is lost."""
        attempt = 0
        while attempt < MAX_ATTEMPTS:
            try:
                self.connect_client()
                if self.client:
                    self.logger.info("Reconnected.")
                    return True
            except Exception as e:
                self.logger.error(e, end=" | ")
            finally:
                attempt += 1
                self.logger.debug(f"Attempt: {attempt}")  # Changed to debug level
                time.sleep(self.wait)
                self.wait = min(self.wait + self.wait_increase, MAX_WAIT)  # Updated to cap the wait time
                self.wait_increase += WAIT_INCREASE

        self.logger.error("Max_attempts reached!")
        self.logger.error("Could not connect with Binance server.")
        self.stop_all()
        return False

    def start_trading(self, tickers_config):
        for ticker_config in tickers_config:
            symbol = ticker_config.ticker
            bar_length = ticker_config.bar_length
            lookback_days = ticker_config.lookback_days
            self.logger.debug(f"Processing ticker config for symbol: {symbol}")
            self.logger.debug(f"Bar length: {bar_length}")
            self.logger.debug(f"Lookback days: {lookback_days}")

            self.get_most_recent(symbol, bar_length, lookback_days)
            self.logger.debug(f"Most recent data fetched for symbol: {symbol}")
            self.twm.start_kline_socket(callback=self.stream_candles, symbol=symbol, interval=bar_length)
            self.logger.debug(f"Started kline socket for symbol: {symbol} with interval: {bar_length}")

    def get_most_recent(self, symbol, interval, lookback_days):
        """Fetches the most recent trading data."""
        now = datetime.utcnow()
        past = str(now - timedelta(days=lookback_days))

        bars = self.client.get_historical_klines(symbol=symbol, interval=interval, start_str=past, limit=1000)
        df = pd.DataFrame(bars, columns=["Open Time", "Open", "High", "Low", "Close", "Volume",
                                         "Close Time", "Quote Asset Volume", "Number of Trades",
                                         "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"])

        df["Date"] = pd.to_datetime(df["Open Time"], unit="ms")
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.set_index("Date", inplace=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        df["Complete"] = [True for _ in range(len(df) - 1)] + [False]

        self.data[symbol] = df

    def stream_candles(self, msg):
        """Streams candle data."""
        # Extract the symbol (ticker) from the message
        symbol = msg["s"]
        # Extract the start time for the current candle
        start_time = pd.to_datetime(msg["k"]["t"], unit="ms")
        # Prepare the data for the current candle
        start_time = pd.to_datetime(msg["k"]["t"], unit="ms")
        # Prepare the data for the current candle
        data = {
            "Open": float(msg["k"]["o"]),
            "High": float(msg["k"]["h"]),
            "Low": float(msg["k"]["l"]),
            "Close": float(msg["k"]["c"]),
            "Volume": float(msg["k"]["v"]),
            "Complete": msg["k"]["x"]
        }

        # Update the DataFrame for the given symbol with the new data
        self.data[symbol].loc[start_time] = data
        # If the current candle is complete, set the data completed event for the given symbol
        if data["Complete"]:
            self.data_completed_events[symbol].set()