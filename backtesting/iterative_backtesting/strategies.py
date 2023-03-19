from abc import ABC, abstractmethod


class Strategy(ABC):
    max_periods = None

    @abstractmethod
    def execute(self, data):
        pass

    def set_params(self, **kwargs):
        pass


class SMA(Strategy):
    def __init__(self, **kwargs):
        self.sma_s = int(kwargs["sma_s"])
        self.sma_l = int(kwargs["sma_l"])
        self.max_periods = max(self.sma_s, self.sma_l)
        print(f"Constructor SMA: {self.sma_s=}")
        print(f"Constructor SMA: {self.sma_l=}")

    def execute(self, data):
        signal = ""
        df = data.copy()
        df["EMA_S"] = df["Close"].rolling(self.sma_s).mean()
        df["EMA_L"] = df["Close"].rolling(self.sma_l).mean()

        if df["EMA_S"].iloc[-1] > df["EMA_L"].iloc[-1]:
            signal = "Buy"

        if df["EMA_S"].iloc[-1] < df["EMA_L"].iloc[-1]:
            signal = "Sell"

        return signal

    def set_params(self, **kwargs):
        self.sma_s = kwargs["sma_s"]
        self.sma_l = kwargs["sma_l"]


class EMA(Strategy):

    def __init__(self, **kwargs):
        self.ema_s = kwargs["ema_s"]
        self.ema_l = kwargs["ema_l"]
        print(f"Constructor EMA: {self.ema_s=}")
        print(f"Constructor EMA: {self.ema_l=}")
        self.max_periods = max(self.ema_s, self.ema_l)

    def execute(self, data):
        signal = ""
        df = data.copy()
        ema_s_col = df["Close"].ewm(span=self.ema_s, min_periods=self.ema_s).mean()
        ema_l_col = df["Close"].ewm(span=self.ema_l, min_periods=self.ema_l).mean()

        #print(df.head())
        #print("="*100)
        #print(ema_s_col)
        #print(ema_l_col)
        #print(f"EMA_S last: {ema_s_col.iloc[-1]}, EMA_L last: {ema_l_col.iloc[-1]}")
        if ema_s_col.iloc[-1] > ema_l_col.iloc[-1]:
            signal = "Buy"
        if ema_s_col.iloc[-1] < ema_l_col.iloc[-1]:
            signal = "Sell"

        return signal

    def execute_(self, data):
        signal = ""
        #df = data.copy()
        data["ema_s"] = data["Close"].ewm(span=self.ema_s, min_periods=self.ema_s).mean()
        data["ema_l"]  = data["Close"].ewm(span=self.ema_l, min_periods=self.ema_l).mean()

        #data["position"] = 0
        # print(df.head())
        # print("="*100)
        # print(ema_s_col)
        # print(ema_l_col)
        # print(f"EMA_S last: {ema_s_col.iloc[-1]}, EMA_L last: {ema_l_col.iloc[-1]}")
        if  data["ema_s"].iloc[-1] > data["ema_l"].iloc[-1]:
            data["signal"].iloc[-1] = 1
        if data["ema_s"].iloc[-1] < data["ema_l"].iloc[-1]:
            data["signal"].iloc[-1] = 0

        #return signal

    def set_params(self, **kwargs):
        self.ema_s = kwargs["ema_s"]
        self.ema_l = kwargs["ema_l"]


class MACD(Strategy):
    def __init__(self, **kwargs):
        self.ema_s = int(kwargs["ema_s"])
        self.ema_l = int(kwargs["ema_l"])
        self.signal_sw = int(kwargs["signal_mw"])

        self.max_periods = max(self.ema_s, self.ema_l, self.signal_sw)
        print(f"Constructor MACD: {self.ema_s=}")
        print(f"Constructor MACD: {self.ema_l=}")
        print(f"Constructor MACD: {self.signal_sw=}")

    def execute(self, data):
        df = data.copy()
        # Calculate EMA and MACD
        ema_s_col = df["Close"].ewm(span=self.ema_s, min_periods=self.ema_s).mean()
        ema_l_col = df["Close"].ewm(span=self.ema_l, min_periods=self.ema_l).mean()
        macd = ema_s_col - ema_l_col
        signal_col = macd.ewm(span=self.signal_sw, min_periods=self.signal_sw).mean()
        df.dropna(inplace=True)

        # Determine signal
        if macd.iloc[-1] > signal_col.iloc[-1]:
            signal = "Buy"
        elif macd.iloc[-1] < signal_col.iloc[-1]:
            signal = "Sell"
        else:
            signal = ""
        return signal

    def set_params(self, **kwargs):
        self.ema_s = kwargs["ema_s"]
        self.ema_l = kwargs["ema_l"]
        self.signal_sw = kwargs["signal_mw"]


class BB(Strategy):
    def __init__(self, **kwargs):
        self.sma = int(kwargs["sma"])
        self.dev = float(kwargs["dev"])
        print(f"Constructor BB: {self.sma=}")
        print(f"Constructor BB: {self.dev=}")

        self.max_periods = max(self.sma, self.dev)

    def execute(self, data):
        df = data.copy()

        ma = df['Close'].ewm(span=self.sma, min_periods=self.sma).mean()
        bb_up = ma + self.dev * df['Close'].rolling(self.sma).std(ddof=0)
        bb_down = ma - self.dev * df['Close'].rolling(self.sma).std(ddof=0)

        if bb_up.iloc[-1] > df.Close.iloc[-1]:
            signal = "Sell"
        elif bb_down.iloc[-1] < df.Close.iloc[-1]:
            signal = "Buy"
        else:
            signal = ""
        return signal

    def set_params(self, **kwargs):
        self.sma = kwargs["sma"]
        self.dev = kwargs["dev"]


class RSI(Strategy):
    def __init__(self, **kwargs):
        self.periods = int(kwargs["periods"])
        self.rsi_lower = int(kwargs["rsi_lower"])
        self.rsi_upper = int(kwargs["rsi_upper"])
        print(f"Constructor RSI: {self.periods=}")
        print(f"Constructor RSI: {self.rsi_lower=}")
        print(f"Constructor RSI: {self.rsi_upper=}")

        self.max_periods = max(self.periods, self.rsi_upper, self.rsi_lower
                               )
    def execute(self, data):
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
            signal = "Buy"
        elif rsi.iloc[-1] >= self.rsi_upper:
            signal = "Sell"
        else:
            signal = ""

        return signal

    def set_params(self, **kwargs):
        self.periods = kwargs["periods"]
        self.rsi_lower = kwargs["rsi_lower"]
        self.rsi_upper = kwargs["rsi_upper"]


class SO(Strategy):
    def __init__(self, **kwargs):
        self.periods = int(kwargs["periods"])
        self.d_mw = int(kwargs["d_mw"])
        print(f"Constructor SO: {self.periods=}")
        print(f"Constructor SO: {self.d_mw=}")
        self.max_periods = max(self.periods, self.d_mw)

    def execute (self, data):
        df = data.copy()

        # Calculate the %K and %D values
        low_min = df["Low"].rolling(self.periods).min()
        high_max = df["High"].rolling(self.periods).max()
        df["%K"] = (df["Close"] - low_min) / (high_max - low_min) * 100
        df["%D"] = df["%K"].rolling(self.d_mw).mean()

        # Generate signals based on the %K and %D values
        if df["%K"].iloc[-1] > df["%D"].iloc[-1] and df["%K"].iloc[-2] < df["%D"].iloc[-2]:
            signal = "Buy"
        elif df["%K"].iloc[-1] < df["%D"].iloc[-1] and df["%K"].iloc[-2] > df["%D"].iloc[-2]:
            signal = "Sell"
        else:
            signal = ""

        return signal

    def set_params(self, **kwargs):
        self.periods = kwargs["periods"]
        self.d_mw = kwargs["d_mw"]


class BacktesterBase:
    def __init__(self, units, initial_balance, symbol, strategy):
        self.units = units
        self.initial_balance = initial_balance
        self.symbol = symbol
        self.strategy = strategy

    def buy_instrument(self):
        self.strategy.execute()
        # Implementierung für den Kauf eines Instruments

    def sell_instrument(self):
        self.strategy.execute()
        # Implementierung für den Verkauf eines Instruments

    def close_position(self):
        self.strategy.execute()
        # Implementierung zum Schließen einer Position

# Executor-Klasse für den Backtester
class StrategyExecutor:
    def __init__(self, strategy_name, units, initial_balance, symbol):
        self.units = units
        self.initial_balance = initial_balance
        self.symbol = symbol
        self.strategy = self.get_strategy(strategy_name)

    def get_strategy(self, strategy_name):
        if strategy_name == 'SMA':
            return SMA()
        elif strategy_name == 'EMA':
            return EMA()
        elif strategy_name == 'MACD':
            return MACD()
        elif strategy_name == 'RSI':
            return RSI()
        else:
            raise ValueError(f'Invalid strategy name: {strategy_name}')

    def run_backtest(self):
        backtester = BacktesterBase(self.units, self.initial_balance, self.symbol, self.strategy)
        # Implementierung für den Backtest