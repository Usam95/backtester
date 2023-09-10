
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(".."))
from data_loader import load_data
from logger import Logger

logger = Logger().get_logger()


class FeatureEngineer:
    def __init__(self, data):
        """Initialize the FeatureEngineer with a dataset."""
        self.data = data.copy().astype(float)
        self._validate_data()
        self.add_features()

    def _validate_data(self):
        """Validate the input data for necessary columns."""
        required_columns = ['Close', 'Volume', 'High', 'Low']

        missing_columns = [col for col in required_columns if col not in self.data.columns]

        if missing_columns:
            raise ValueError(f"Input data is missing the following columns: {', '.join(missing_columns)}")

    def add_features(self):
        """Add various technical indicators to the dataset."""
        for n in [10, 30, 200]:
            self._add_ma(n)
            self._add_ema(n)
            self._add_rsi(n)
            self._add_sto(n)
            self._add_mom(n)
            self._add_roc(n)
        self._add_obv()
        self._add_returns()
        self._add_rolling_cum_returns()
        self._add_rolling_cum_range()
        self._add_day_of_week()
        self. _calculate_range()

    def _add_returns(self):
        """Calculate and add the log returns to the dataset."""
        self.data["returns"] = np.log(self.data.Close / self.data.Close.shift())

    def _add_ma(self, n):
        """Add moving average over a window of n periods."""
        self.data[f'MA_{n}'] = self.data['Close'].rolling(n, min_periods=n).mean()

    def _add_obv(self):
        """Add On-Balance Volume indicator."""
        self.data["OBV"] = (np.sign(self.data["Close"].diff()) * self.data["Volume"]).fillna(0).cumsum()

    def _add_ema(self, n):
        """Add Exponential Moving Average over a window of n periods."""
        self.data[f'EMA_{n}'] = self.data['Close'].ewm(span=n, min_periods=n).mean()

    def _add_mom(self, n):
        """Add Momentum indicator over a window of n periods."""
        self.data[f'Momentum_{n}'] = self.data['Close'].diff(n)

    def _add_rsi(self, n):
        """Add Relative Strength Index over a window of n periods."""
        delta = self.data["Close"].diff()
        u = (delta * (delta > 0)).fillna(0)
        d = (-delta * (delta < 0)).fillna(0)
        rs = u.ewm(span=n, adjust=False).mean() / d.ewm(span=n, adjust=False).mean()
        self.data[f'RSI_{n}'] = 100 - 100 / (1 + rs)

    def _add_roc(self, n):
        """Add Rate of Change over a window of n periods."""
        M = self.data["Close"].diff(n - 1)
        N = self.data["Close"].shift(n - 1)
        self.data[f'ROC_{n}'] = ((M / N) * 100)

    def _add_sto(self, n):
        """Add Stochastic Oscillator and Stochastic Oscillator Divergence."""
        stock_str = f'STOCK_{n}'
        stod_str = f'STOD_{n}'
        self.data[stock_str] = ((self.data["Close"] - self.data["Low"].rolling(n).min()) /
                               (self.data["High"].rolling(n).max() - self.data["Low"].rolling(n).min())) * 100
        self.data[stod_str] = self.data[stock_str].rolling(3).mean()

    def _add_target(self, short_mavg=10, long_mavg=60):
        """Add target signal based on moving averages crossover."""
        self.data['short_mavg'] = self.data['Close'].rolling(window=short_mavg, min_periods=1, center=False).mean()
        self.data['long_mavg'] = self.data['Close'].rolling(window=long_mavg, min_periods=1, center=False).mean()
        self.data['signal'] = np.where(self.data['short_mavg'] > self.data['long_mavg'], 1, 0)
        return self.data

    def _add_rolling_cum_returns(self):
        """Add rolling cumulative returns."""
        self.data["Roll_Rets"] = self.data["returns"].rolling(window=30).sum()

    def _add_rolling_cum_range(self):
        """Add rolling average range."""
        self.data["Avg_Range"] = self.data["High"] / self.data["Low"] - 1

    def _add_day_of_week(self):
        """Add day of the week."""
        self.data["DOW"] = self.data.index.dayofweek

    def _calculate_range(self):
        """Calculate the range."""
        self.data["Range"] = self.data["High"] / self.data["Low"] - 1


if __name__ == "__main__":
    symbol = "XRPUSDT"
    data = load_data(symbol)
    freq = f"{720}min"
    data_resampled = data.resample(freq).last().dropna().iloc[:-1].copy()
    preparer = FeatureEngineer(data_resampled)
    print(preparer.data.columns)
