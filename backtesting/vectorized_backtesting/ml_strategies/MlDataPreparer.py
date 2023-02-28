import pandas as pd
import os

import numpy as np
import tensorflow as tf
#from tensorflow.keras import layers
#import matplotlib.pyplot as plt
#
from sklearn.model_selection import train_test_split

import sklearn.mixture as mix
from ta.momentum import RSIIndicator

from sklearn.preprocessing import MinMaxScaler

class MlDataPreparer:

    def __init__(self):
        self.hist_data_folder = "../../../hist_data"
    
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
        #self.data = self.data.resample(freq).last()

    # Add Features
    def add_features(self):
        
        df = self.data.copy()
        window = 50

        df["sma"] = df.Close.rolling(window).mean() - df.Close.rolling(150).mean()
        df["boll"] = (df.Close - df.Close.rolling(window).mean()) / df.Close.rolling(window).std()
        df["min"] = df.Close.rolling(window).min() / df.Close - 1
        df["max"] = df.Close.rolling(window).max() / df.Close - 1
        df["mom"] = df["returns"].rolling(3).mean()
        df["vol"] = df["returns"].rolling(window).std()

        # Add RSI
        rsi = RSIIndicator(close=df["Close"], window=14).rsi()
        df["rsi"] = rsi
        df["rsi_ret"] = df["rsi"] / df["rsi"].shift(1)

        # Add Moving Average
        df["ma_12"] = df["Close"].rolling(window=12).mean()
        df["ma_21"] = df["Close"].rolling(window=21).mean()

        # Day of Week
        df["dow"] = df.index.dayofweek

        df["range"] = df["High"] / df["Low"] - 1

        # Rolling Cumulative Returns
        df["roll_rets"] = df["returns"].rolling(window=30).sum()

        # Rolling Cumulative Range
        df["avg_range"] = df["range"].rolling(window=30).mean()

        t_steps = [1, 2, 5, 7, 9]
        t_features = ["returns", "range", "rsi_ret"]
        for ts in t_steps:
            for tf in t_features:
                df[f"{tf}_t{ts}"] =df[tf].shift(ts)

        lags = 5
        cols = []
        features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]

        for f in features:
            for lag in range(1, lags + 1):
                col = "{}_lag_{}".format(f, lag)
                df[col] = df[f].shift(lag)
                cols.append(col)
        df.dropna(inplace = True)
        self.data = df
        
    def add_ema(self, n): 
        ema_str = 'EMA_' + str(n)
        self.data[ema_str] = pd.Series(self.data['Close'].ewm(span=n, min_periods=n).mean(), name=ema_str)

    def add_mom(self, n):
        mom_str = 'Momentum_' + str(n)
        self.data[mom_str] = pd.Series(self.data['Close'].diff(n), name=mom_str)

    # calculation of relative strength index
    def add_rsi(self, period):
        delta = self.data["Close"].diff()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        u[u.index[period-1]] = np.mean(u[:period]) # first value is sum of avg gains
        #u = u.drop(u.index[:(period-1)])
        d[d.index[period-1]] = np.mean(d[:period]) # first value is sum of avg losses
        #d = d.drop(d.index[:(period-1)])
        rs = u.ewm(com=period-1, adjust=False).mean() / \
        d.ewm(com=period-1, adjust=False).mean()
        rsi_str = 'RSI_' + str(period)
        self.data[rsi_str] = 100 - 100 / (1 + rs)

    # calculation of rate of change
    def add_ros(self, n):
        M = self.data["Close"].diff(n - 1)
        N = self.data["Close"].shift(n - 1)
        roc_str = 'ROC_' + str(n)
        self.data[roc_str] = pd.Series(((M / N) * 100), name=roc_str)

    def add_stok(self, n: int) -> None:
        stock_str = 'STOCK_' + str(n)
        self.data[stock_str] = ((self.data["Close"] - self.data["Low"].rolling(n).min())
                                / (self.data["High"].rolling(n).max() - self.data["Low"].rolling(n).min())) * 100

    def add_stod(self, n: int) -> None:
        stod_str = 'STOD_' + str(n)
        stock_str = 'STOCK_' + str(n)
        self.data[stock_str] = ((self.data["Close"] - self.data["Low"].rolling(n).min())
                               / (self.data["High"].rolling(n).max() - self.data["Low"].rolling(n).min())) * 100
        self.data[stod_str] = self.data[stock_str].rolling(3).mean()

    def add_label(self):
        self.data["returns"] = np.log(self.data.Close / self.data.Close.shift())
        self.data["dir"] = np.where(self.data["returns"].shift(-1) > 0, 1, 0)
        
    def convert_to_floats(self):
        self.data = self.data.astype(float)

    def remove_columns(self):
        self.data.drop(columns=["Open", "High", "Low"], inplace=True)
        
    def split(self):
        self.X = self.data.loc[:, self.data.columns != "dir"].values
        self.y = self.data.loc[:, "dir"].values

    def scale(self):
        scaler = MinMaxScaler()
        self.data = pd.DataFrame(scaler.fit_transform(self.data), columns=self.data.columns, index=self.data.index)
        
    def create_datasets(self, ml, verbose=True):
        """
        Splits original dataset into training, validation and testing datasets.
        - training dataset: 80% of original dataset
        - validation dataset: 10% of original dataset
        - testing dataset: 10% of original dataset
        """
        # added these two lines for ml_strategies pipeline
        if ml:
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X, self.y, test_size=0.3,
                                                                                      random_state=42, shuffle=False)
            print("ok")
            if verbose:
                print("DL model dataset parameters:\n")
                print(f"Length of training dataset:\t{len( self.X_train)}")
                print(f"Length of validation dataset:\t{int(len( self.X_valid))}")
                print("=" * 100)
        else:

            X_train, X_valid, y_train, y_valid = train_test_split(self.X, self.y, test_size=0.3,
                                                                  random_state=42, shuffle=False)
            self.training_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(16).prefetch(tf.data.AUTOTUNE)
            valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

            val_dataset_length = len(X_valid)
            val_test_idx = int(val_dataset_length * 0.5)

            # Split validation dataset into validation and testing datasets
            self.valid_dataset = valid_dataset.take(val_test_idx).batch(16).prefetch(tf.data.AUTOTUNE)
            self.testing_dataset = valid_dataset.skip(val_test_idx).prefetch(tf.data.AUTOTUNE)

            if verbose: 
                print("DL model dataset parameters:\n")
                print(f"Length of training dataset:\t{len(X_train)}")
                print(f"Length of validation dataset:\t{int(len(X_valid)*0.5)}")
                print(f"Length of testing dataset:\t{int(len(X_valid)*0.5)}")
                print("=" * 100)

    def windowing(self, window_size=20):
        
        # Window the data but only take those with the specified size
        self.training_dataset = self.training_dataset.window(window_size, shift=1, drop_remainder=True)
        self.training_dataset = self.training_dataset.flat_map(lambda window: window.batch(5))
        self.training_dataset = self.training_dataset.map(lambda window: (window[:-1], window[-1:]))

        # vself.valid_dataset = self.valid_dataset.window(window_size, shift=1, drop_remainder=True)
        # dataset = dataset.flat_map(lambda window: window.batch(5))
        # dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
        # self.testing_dataset = valid_dataset.skip(val_test_idx).prefetch(tf.data.AUTOTUNE)
        
    def sliding_window(self, timestemps=7):
        
        x = []
        y = []
        
        for i in range(len(self.X)-timestemps):
            _x = self.X[i:(i+timestemps)]
            _y = self.y[i+timestemps-1]
            
            x.append(_x)
            y.append(_y)
        
        return np.array(x), np.array(y)
        
    def get_datasets(self, symbol, ml):
        self.load_data(symbol)
        self.add_label()

        self.add_features()

        #self.remove_columns()
        self.convert_to_floats()
        self.scale()
        self.split()
        
        self.create_datasets(ml)
        if not ml:
            self.X, self.y = self.sliding_window()
        

if __name__ == "__main__":

    preparer = MlDataPreparer()
    symbol = "XRPUSDT"
    preparer.get_datasets(symbol, ml=True)

    for period in range(10, 200, 10):
        preparer.add_ema(period)
    for period in range(10, 200, 30):
        preparer.add_mom(period)
    for period in range(10, 200, 20):
        preparer.add_ros(period)
    for period in range(10, 200, 20):
        preparer.add_stod(period)
    for period in range(10, 200, 20):
        preparer.add_rsi(period)

    preparer.remove_columns()
    print(preparer.data.columns)
    print(preparer.data.shape)