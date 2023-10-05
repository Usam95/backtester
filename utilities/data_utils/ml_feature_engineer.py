
import os
import sys
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(".."))
from .data_loader import load_data
from ..logger import Logger

logger = Logger().get_logger()


class FeatureEngineer:
    def __init__(self, data: pd.DataFrame, periods: list):
        """Initialize the FeatureEngineer with a dataset."""
        self.data = data
        self.periods = periods

    def add_features(self):
        """Add various technical indicators to the dataset."""
        for n in self.periods:
            self._add_ma(n)
            self._add_ema(n)
            self._add_rsi(n)
            self._add_sto(n)
            self._add_mom(n)
            self._add_roc(n)
            self._add_stos(n)
            self._add_stomom(n)
            self._add_storoc(n)
            self._add_stoch_rsi(n)
            self._add_sto_cross(n)
        self._add_obv()
        self._add_returns()
        self._add_rolling_cum_returns()
        self._add_rolling_cum_range()
        self._add_day_of_week()
        self._calculate_range()

        return self.data

    def _add_returns(self):
        """Calculate and add the log returns to the dataset."""
        self.data["Returns"] = np.log(self.data.Close / self.data.Close.shift())
        cum_returns = np.exp(self.data["Returns"].cumsum())
        print(f"Id add_returns..")
        print(cum_returns.tail())

    def _add_ma(self, n):
        """Add moving average over a window of n periods."""
        self.data[f'MA_{n}'] = self.data['Close'].rolling(n, min_periods=n).mean()

    def _add_obv(self):
        """Add On-Balance Volume indicator."""
        self.data["OBV"] = (np.sign(self.data["Close"].diff()) * self.data["Volume"]).fillna(0).cumsum()
        # Transform OBV to handle negative values and reduce magnitude
        self.data['OBV'] = np.sign(self.data['OBV']) * np.log1p(np.abs(self.data['OBV']))

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

    def _add_stos(self, n):
        """Add Slow Stochastic Oscillator."""
        stock_str = f'STOCK_{n}'
        self.data[f'STOS_{n}'] = self.data[stock_str].rolling(3).mean()

    def _add_stomom(self, n):
        """Add Stochastic Oscillator Momentum."""
        stock_str = f'STOCK_{n}'
        self.data[f'STOMOM_{n}'] = self.data[stock_str] - self.data[stock_str].shift(n)

    def _add_storoc(self, n):
        """Add Stochastic Oscillator Rate of Change."""
        stock_str = f'STOCK_{n}'
        self.data[f'STOROC_{n}'] = self.data[stock_str].pct_change(periods=n) * 100

    def _add_stoch_rsi(self, n):
        """Add Stochastic RSI."""
        delta = self.data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(n).mean()
        avg_loss = loss.rolling(n).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        self.data[f'StochRSI_{n}'] = ((rsi - rsi.rolling(n).min()) /
                                      (rsi.rolling(n).max() - rsi.rolling(n).min())) * 100

    def _add_sto_cross(self, n):
        """Add Stochastic Oscillator Cross."""
        stock_str = f'STOCK_{n}'
        stod_str = f'STOD_{n}'
        self.data[f'STOCROSS_{n}'] = np.where(self.data[stock_str] > self.data[stod_str], 1, -1)

    def add_target(self, config):
        strategy_target = config.target_conf.target
        """
        Calculate a target variable based on a given strategy.

        Args:
        - data (pd.DataFrame): The input data with historical prices and indicators.
        - target (str): The name of the target strategy to calculate. Options: 'Simple', 'MA_Relative', 'Momentum', 'ROC', 'RSI_Cross'.

        Returns:
        - pd.DataFrame: The dataframe with a new column 'Signal' containing the binary target.
        """

        if strategy_target == 'Simple':
            """
            Compare the current close price to the next close price. 
            If the next close is higher, then it's an "up" movement (label=1). 
            Otherwise, it's a "down" movement (label=0).
            """
            self.data['Signal'] = np.where(self.data['Close'].shift(-1) > self.data['Close'], 1, 0)

        elif strategy_target == 'MA_Relative':
            """
            Compare the current close price to its short-term moving average. 
            If the close is above the moving average and rising, it's an "up" (label=1). 
            If below and falling, it's "down" (label=0).
            """
            self.data['Signal'] = np.where((self.data['Close'] > self.data['MA_10']) & (self.data['Close'].shift(-1) > self.data['Close']), 1, 0)

        elif strategy_target == 'Momentum':
            """
            If the momentum is positive and rising, it's an "up" (label=1). 
            If negative and falling, it's "down" (label=0)
            """
            self.data['Signal'] = np.where((self.data['Momentum_10'] > 0) & (self.data['Momentum_10'].shift(-1) > self.data['Momentum_10']), 1, 0)

        elif strategy_target == 'ROC':
            """
            If the rate of change (ROC) is positive and increasing, label as "up" (label=1). 
            If ROC is negative and decreasing, label as "down" (label=0).
            """
            self.data['Signal'] = np.where((self.data['ROC_10'] > 0) & (self.data['ROC_10'].shift(-1) > self.data['ROC_10']), 1, 0)

        elif strategy_target == 'RSI_Cross':
            """
             If RSI crosses above a predefined threshold (like 30), suggesting moving out from an oversold region, label as "up" (label=1). 
             If RSI crosses below a certain level (like 70), indicating moving into an overbought territory, label as "down" (label=0).
            """
            self.data['Signal'] = np.where((self.data['RSI_10'] > 30) & (self.data['RSI_10'].shift(1) <= 30), 1,
                                            np.where((self.data['RSI_10'] < 70) & (self.data['RSI_10'].shift(1) >= 70), 0, np.nan))

        elif strategy_target == 'Returns_based':
            """
            Predict if the returns of the next period are positive.
            """
            self.data['Signal'] = np.where(self.data['Returns'].shift(-1) > 0, 1, 0)

        elif strategy_target == 'Exceed_Avg_Returns':
            """
            Predict if the returns of the next period are positive.
            """
            N = config.target_conf.N or 10
            self.data['Signal'] = np.where(self.data['Returns'].shift(-1) > self.data['Returns'].rolling(N).mean(), 1, 0)

        elif strategy_target == 'Volatility_Breakout':
            """
            Predict if the absolute returns (as a measure of volatility) of the next period will exceed the average absolute returns of the past N periods.
            This can be an indication of a breakout or a major market event.
            """
            N = config.target_conf.N or 10
            self.data['Signal'] = np.where(abs(self.data['Returns'].shift(-1)) > abs(self.data['Returns'].rolling(N).mean()), 1, 0)

        elif strategy_target == 'Exceed_Threshold':
            """
            Set a threshold (e.g., 1% or 0.01 as a fraction) and predict if the returns for the next period will exceed this threshold.
            """
            threshold = config.target_conf.threshold or 0.01
            self.data['Signal'] = np.where(self.data['Returns'].shift(-1) > threshold, 1, 0)

        elif strategy_target == 'Consecutive_Increases':
            """
            Predict if the next period's returns will mark the third consecutive increase in returns. This can capture trending behavior.
            """
            self.data['Signal'] = np.where((self.data['Returns'] > 0)
                                            & (self.data['Returns'].shift(1) > 0)
                                            & (self.data['Returns'].shift(2) > 0)
                                            & (self.data['Returns'].shift(3) < 0), 1, 0)

        else:
            raise ValueError(f"Target strategy '{strategy_target}' not recognized. Please provide the valid strategy.")

        return self.data

    def _add_rolling_cum_returns(self):
        """Add rolling cumulative returns."""
        self.data["Roll_Rets"] = self.data["Returns"].rolling(window=30).sum()

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
