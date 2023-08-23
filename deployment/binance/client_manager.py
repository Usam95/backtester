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
        self.logging = Logger().get_logger()
        self.logging.debug("Binance client was initialised successfully.")
        #self.connect_client()
        #self.start_trading()


    def connect_client(self):
        """Establish a connection with the Binance client."""
        credentials = (testnet_api_key, testnet_secret_key) if self.testnet else (api_key, secret_key)
        try:
            self.client = Client(api_key=credentials[0], api_secret=credentials[1], tld="com", testnet=self.testnet)
            self.twm.start()
            self.logging.info("Binance client was started successfully.")
        except Exception as e:
            self.logging.error(f"Error occurred during connection establishment with server: {e}")
            self.logging.error(f"Exiting...")
            sys.exit(1)

    def client_is_connected(self, reconnect=True, verbose=True) -> bool:
        """Check if the client is connected to Binance."""
        try:
            if len(self.client.ping()) == 0:
                if verbose:
                    self.logging.info("Connection is ok.")
                return True
        except Exception as e:
            if verbose:
                self.logging.error(f"Connection with Binance server was lost due to: {e}. Trying to reconnect..")
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
                    self.logging.info("Reconnected.")
                    return True
            except Exception as e:
                self.logging.error(e, end=" | ")
            finally:
                attempt += 1
                self.logging.debug(f"Attempt: {attempt}")  # Changed to debug level
                time.sleep(self.wait)
                self.wait = min(self.wait + self.wait_increase, MAX_WAIT)  # Updated to cap the wait time
                self.wait_increase += WAIT_INCREASE

        self.logging.error("Max_attempts reached!")
        self.logging.error("Could not connect with Binance server.")
        self.stop_all()
        return False

    def start_trading(self, tickers_config):
        for ticker_config in tickers_config:
            symbol = ticker_config.ticker
            bar_length = ticker_config.bar_length
            lookback_days = ticker_config.lookback_days
            self.logging.info(f"Processing ticker config for symbol: {symbol}")
            self.logging.info(f"Bar length: {bar_length}")
            self.logging.info(f"Lookback days: {lookback_days}")

            self.get_most_recent(symbol, bar_length, lookback_days)
            self.logging.info(f"Most recent data fetched for symbol: {symbol}")
            self.twm.start_kline_socket(callback=self.stream_candles, symbol=symbol, interval=bar_length)
            self.logging.info(f"Started kline socket for symbol: {symbol} with interval: {bar_length}")

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

        # Update the self.data structure for the symbol and interval
        if symbol not in self.data:
            self.data[symbol] = {}
        self.data[symbol][interval] = df

    def stream_candles(self, msg):
        """Streams candle data."""
        # Extract the symbol (ticker) from the message
        symbol = msg["s"]
        # Extract the start time for the current candle
        interval = msg["k"]["i"]  # Extracting the interval from the message
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

        # Ensure the symbol and interval are in self.data
        if symbol not in self.data:
            self.data[symbol] = {}
        if interval not in self.data[symbol]:
            self.data[symbol][interval] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Complete"])

        # Update the self.data structure for the symbol and interval
        self.data[symbol][interval].loc[start_time] = data
        # If the current candle is complete, set the data completed event for the given symbol
        if data["Complete"]:
            self.data_completed_events[symbol].set()
            self.logging.info(f"Set data_completed_events {self.data_completed_events[symbol]} for {symbol}..")