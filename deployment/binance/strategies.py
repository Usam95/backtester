from abc import ABC, abstractmethod
from enum import Enum
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '', '..', '..')))
from utilities.logger import Logger


logger = Logger().get_logger()

class Signal(Enum):
    BUY = 1
    SELL = -1
    NEUTRAL = 0


class StrategyModel(ABC):
    strategy_name = ""

    @abstractmethod
    def get_signal(self, data):
        pass

    @abstractmethod
    def set_params(self, **kwargs):
        pass


class SmaStrategy(StrategyModel):
    def __init__(self, strategy_name, **kwargs):
        self.strategy_name = strategy_name
        self.sma_s = int(kwargs["sma_s"])
        self.sma_l = int(kwargs["sma_l"])
        self.max_periods = max(self.sma_s, self.sma_l)
        print(f"Constructor SMA: {self.strategy_name=}")
        print(f"Constructor SMA: {self.sma_s=}")
        print(f"Constructor SMA: {self.sma_l=}")

    def get_signal(self, data):
        df = data.copy()
        df["SMA_S"] = df["Close"].rolling(self.sma_s).mean()
        df["SMA_L"] = df["Close"].rolling(self.sma_l).mean()

        sma_s = round(df["SMA_S"].iloc[-1], 3)
        sma_l = round(df["SMA_L"].iloc[-1], 3)
        price = round(df["Close"].iloc[-1], 3)
        logger.info(f"SMA Strategy: {sma_s=}, {sma_l=}, {price=}")

        if df["SMA_S"].iloc[-1] > df["SMA_L"].iloc[-1]:
            return Signal.BUY
        elif df["SMA_S"].iloc[-1] < df["SMA_L"].iloc[-1]:
            return self.strategy_name, Signal.SELL
        else:
            return self.strategy_name, Signal.NEUTRAL

    def set_params(self, **kwargs):
        self.sma_s = kwargs["sma_s"]
        self.sma_l = kwargs["sma_l"]


class EmaStrategy(StrategyModel):

    def __init__(self, strategy_name, **kwargs):
        self.strategy_name = strategy_name
        self.ema_s = kwargs["ema_s"]
        self.ema_l = kwargs["ema_l"]
        print(f"Constructor EMA: {self.strategy_name=}")
        print(f"Constructor EMA: {self.ema_s=}")
        print(f"Constructor EMA: {self.ema_l=}")
        self.max_periods = max(self.ema_s, self.ema_l)

    def get_signal(self, data):
        df = data.copy()
        ema_s_col = df["Close"].ewm(span=self.ema_s, min_periods=self.ema_s).mean()
        ema_l_col = df["Close"].ewm(span=self.ema_l, min_periods=self.ema_l).mean()

        ema_s = round(ema_s_col.iloc[-1], 3)
        ema_l = round(ema_l_col.iloc[-1], 3)
        price = round(df["Close"].iloc[-1], 3)
        logger.info(f"EMA Strategy: {ema_s=}, {ema_l=}, {price=}")

        if ema_s_col.iloc[-1] > ema_l_col.iloc[-1]:
            return self.strategy_name, Signal.BUY
        elif ema_s_col.iloc[-1] < ema_l_col.iloc[-1]:
            return self.strategy_name, Signal.SELL
        else:
            return self.strategy_name, Signal.NEUTRAL

    def set_params(self, **kwargs):
        self.ema_s = kwargs["ema_s"]
        self.ema_l = kwargs["ema_l"]


class MacdStrategy(StrategyModel):
    def __init__(self, strategy_name, **kwargs):
        self.strategy_name = strategy_name
        self.ema_s = int(kwargs["ema_s"])
        self.ema_l = int(kwargs["ema_l"])
        self.signal_sw = int(kwargs["signal_mw"])

        self.max_periods = max(self.ema_s, self.ema_l, self.signal_sw)
        print(f"Constructor MACD: {self.ema_s=}")
        print(f"Constructor MACD: {self.ema_l=}")
        print(f"Constructor MACD: {self.signal_sw=}")

    def get_signal(self, data):
        df = data.copy()
        # Calculate EMA and MACD
        ema_s_col = df["Close"].ewm(span=self.ema_s, min_periods=self.ema_s).mean()
        ema_l_col = df["Close"].ewm(span=self.ema_l, min_periods=self.ema_l).mean()
        macd = ema_s_col - ema_l_col
        signal_col = macd.ewm(span=self.signal_sw, min_periods=self.signal_sw).mean()
        df.dropna(inplace=True)

        # Determine signal
        if macd.iloc[-1] > signal_col.iloc[-1]:
            return self.strategy_name, Signal.BUY
        elif macd.iloc[-1] < signal_col.iloc[-1]:
            return self.strategy_name, Signal.SELL
        else:
            return Signal.NEUTRAL

    def set_params(self, **kwargs):
        self.ema_s = kwargs["ema_s"]
        self.ema_l = kwargs["ema_l"]
        self.signal_sw = kwargs["signal_mw"]


class BbStrategy(StrategyModel):
    def __init__(self, strategy_name, **kwargs):
        self.strategy_name = strategy_name
        self.sma = int(kwargs["sma"])
        self.dev = float(kwargs["dev"])
        print(f"Constructor BB: {self.sma=}")
        print(f"Constructor BB: {self.dev=}")

        self.max_periods = max(self.sma, self.dev)

    def get_signal(self, data):
        df = data.copy()

        ma = df['Close'].ewm(span=self.sma, min_periods=self.sma).mean()
        bb_up = ma + self.dev * df['Close'].rolling(self.sma).std(ddof=0)
        bb_down = ma - self.dev * df['Close'].rolling(self.sma).std(ddof=0)

        if bb_up.iloc[-1] > df.Close.iloc[-1]:
            return Signal.SELL
        elif bb_down.iloc[-1] < df.Close.iloc[-1]:
            return self.strategy_name, Signal.BUY
        else:
            return self.strategy_name, Signal.NEUTRAL

    def set_params(self, **kwargs):
        self.sma = kwargs["sma"]
        self.dev = kwargs["dev"]


class RsiStrategy(StrategyModel):
    def __init__(self, strategy_name, **kwargs):
        self.strategy_name = strategy_name
        self.periods = int(kwargs["periods"])
        self.rsi_lower = int(kwargs["rsi_lower"])
        self.rsi_upper = int(kwargs["rsi_upper"])
        print(f"Constructor RSI: {self.periods=}")
        print(f"Constructor RSI: {self.rsi_lower=}")
        print(f"Constructor RSI: {self.rsi_upper=}")

        self.max_periods = max(self.periods, self.rsi_upper, self.rsi_lower)

    def get_signal(self, data):
        df = data.copy()

        # Compute the RSI indicator
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(span=self.periods, min_periods=self.periods).mean()
        avg_loss = loss.ewm(span=self.periods, min_periods=self.periods).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Compute the signal based on RSI values
        if rsi.iloc[-1] <= self.rsi_lower:
            return Signal.BUY
        elif rsi.iloc[-1] >= self.rsi_upper:
            return self.strategy_name, Signal.SELL
        else:
            return self.strategy_name, Signal.NEUTRAL

    def set_params(self, **kwargs):
        self.periods = kwargs["periods"]
        self.rsi_lower = kwargs["rsi_lower"]
        self.rsi_upper = kwargs["rsi_upper"]


class SoStrategy(StrategyModel):
    def __init__(self, strategy_name, **kwargs):
        self.strategy_name = strategy_name
        self.periods = int(kwargs["periods"])
        self.d_mw = int(kwargs["d_mw"])
        print(f"Constructor SO: {self.periods=}")
        print(f"Constructor SO: {self.d_mw=}")
        self.max_periods = max(self.periods, self.d_mw)

    def get_signal(self, data):
        df = data.copy()
        # Calculate the %K and %D values
        low_min = df["Low"].rolling(self.periods).min()
        high_max = df["High"].rolling(self.periods).max()
        df["%K"] = (df["Close"] - low_min) / (high_max - low_min) * 100
        df["%D"] = df["%K"].rolling(self.d_mw).mean()

        # Generate signals based on the %K and %D values
        if df["%K"].iloc[-1] > df["%D"].iloc[-1] and df["%K"].iloc[-2] < df["%D"].iloc[-2]:
            return Signal.BUY
        elif df["%K"].iloc[-1] < df["%D"].iloc[-1] and df["%K"].iloc[-2] > df["%D"].iloc[-2]:
            return self.strategy_name, Signal.SELL
        else:
            return self.strategy_name, Signal.NEUTRAL

    def set_params(self, **kwargs):
        self.periods = kwargs["periods"]
        self.d_mw = kwargs["d_mw"]


# strategies.py

STRATEGY_CLASSES = {
    "ema": EmaStrategy,
    "sma": SmaStrategy,
    "macd": MacdStrategy,
    "rsi": RsiStrategy,
    "so": SoStrategy,
    "bb": BbStrategy
}


def strategy_factory(strategy_name: str, **kwargs) -> StrategyModel:
    """Create an instance of a strategy based on the strategy name and given parameters."""
    return STRATEGY_CLASSES[strategy_name](strategy_name, **kwargs)

