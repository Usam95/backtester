import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class MlDataManager:

    def __init__(self, hist_data_folder="../hist_data"):
        self.hist_data_folder = hist_data_folder

    def get_path(self, symbol):
        symbol_folder = os.path.join(self.hist_data_folder, symbol)
        folder_contents = os.listdir(symbol_folder)
        if f"{symbol}.parquet.gzip" in folder_contents:
            data_path = os.path.join(symbol_folder, f"{symbol}.parquet.gzip")
            return data_path
        elif f"{symbol}.csv" in folder_contents:
            data_path = os.path.join(symbol_folder, f"{symbol}.csv")
            return data_path
        else:
            print(f"ERROR: Could not find any data for {symbol}..")
            return None
        
    def load_data(self, symbol):
        data_path = self.get_path(symbol)
        _, file_extension = os.path.splitext(data_path)
        if file_extension == ".gzip":
            self.data = pd.read_parquet(data_path)
        else:
            self.data = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
    
        self.data.fillna(method="ffill", inplace=True)

    # Add Features
    def add_features(self):
        # add EMA
        self.add_ema(n=10)
        self.add_ema(n=30)
        self.add_ema(n=200)

        # add RSI
        self.add_rsi(n=10)
        self.add_rsi(n=30)
        self.add_rsi(n=200)

        # add STO
        self.add_sto(n=10)
        self.add_sto(n=30)
        self.add_sto(n=200)

        # add MOM
        self.add_sto(n=10)
        self.add_sto(n=30)
        self.add_sto(n=200)

        # add ROC
        self.add_roc(n=10)
        self.add_roc(n=30)
        self.add_roc(n=200)

        # add OBV
        self.add_obv()

    def add_obv(self):
        self.data["OBV"] = (np.sign(self.data["Close"].diff()) * self.data["Volume"]).fillna(0).cumsum()

    def add_ema(self, n): 
        ema_str = 'EMA_' + str(n)
        self.data[ema_str] = pd.Series(self.data['Close'].ewm(span=n, min_periods=n).mean(), name=ema_str)

    def add_mom(self, n):
        mom_str = 'Momentum_' + str(n)
        self.data[mom_str] = pd.Series(self.data['Close'].diff(n), name=mom_str)

    # calculation of relative strength index
    def add_rsi(self, n):
        delta = self.data["Close"].diff() #.dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        u[u.index[n-1]] = np.mean(u[:n]) # first value is sum of avg gains
        d[d.index[n-1]] = np.mean(d[:n]) # first value is sum of avg losses
        rs = u.ewm(com=n-1, adjust=False).mean() / d.ewm(com=n-1, adjust=False).mean()
        rsi_str = 'RSI_' + str(n)
        self.data[rsi_str] = 100 - 100 / (1 + rs)

    # calculation of rate of change
    def add_ros(self, n):
        M = self.data["Close"].diff(n - 1)
        N = self.data["Close"].shift(n - 1)
        roc_str = 'ROC_' + str(n)
        self.data[roc_str] = pd.Series(((M / N) * 100), name=roc_str)

    def add_sto(self, n: int) -> None:
        stod_str = 'STOD_' + str(n)
        stock_str = 'STOCK_' + str(n)
        self.data[stock_str] = ((self.data["Close"] - self.data["Low"].rolling(n).min())
                               / (self.data["High"].rolling(n).max() - self.data["Low"].rolling(n).min())) * 100
        self.data[stod_str] = self.data[stock_str].rolling(3).mean()

    def add_target(self):
        # Initialize the `signals` DataFrame with the `signal` column
        # datas['PriceMove'] = 0.0
        # Create short simple moving average over the short window
        self.data['short_mavg'] = self.data['Close'].rolling(window=10, min_periods=1, center=False).mean()
        # Create long simple moving average over the long window
        self.data['long_mavg'] = self.data['Close'].rolling(window=60, min_periods=1, center=False).mean()
        # Create signals
        self.data['signal'] = np.where(self.data['short_mavg'] > self.data['long_mavg'], 1, 0)

    def convert_to_floats(self):
        self.data = self.data.astype(float)

    def remove_columns(self):
        self.data.drop(columns=["Open", "High", "Low", "Volume"], inplace=True)
        
    def split(self):
        self.X = self.data.loc[:, self.data.columns != "signal"]
        self.y = self.data.loc[:, "signal"]

    def scale(self):
        scaler = MinMaxScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns, index=self.data.index)

if __name__ == "__main__":

    preparer = MlDataManager()
    symbol = "XRPUSDT"

    print(preparer.data.columns)
    print(preparer.data.shape)

