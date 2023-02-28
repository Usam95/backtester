import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from scipy.linalg import svd
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MaxNLocator

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


class MlDataManager:
    def __init__(self, hist_data_folder="../hist_data", training=True):
        self.hist_data_folder = hist_data_folder
        self.training = training
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.y_pred = None
        self.features = None

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

    def set_data(self, data):
        """
        Set data:
        :param Pandas DataFrame data: Historical data of cryptocurrency.
        """
        self.data = data.copy()

    def load_data(self, symbol):
        data_path = self.get_path(symbol)
        if os.path.exists(data_path):
            _, file_extension = os.path.splitext(data_path)
            if file_extension == ".gzip":
                self.data = pd.read_parquet(data_path)
            else:
                self.data = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
            self.data.fillna(method="ffill", inplace=True)
        else:
            print(f"ERROR: The path {data_path} does not exist.")

    def down_sample(self, freq, parse_dates=False):
        freq = "{}min".format(freq)
        volume = self.data["Volume"].resample(freq).sum().iloc[:-1]
        self.data = self.data.loc[:, self.data.columns != "Volume"].resample(freq).last().iloc[:-1]
        self.data["Volume"] = volume
        if not parse_dates:
            self.data.reset_index(drop=True, inplace=True)

    def preprocess_data(self, short_mavg=10, long_mavg=60, transform="standardize", freq=60):

        if self.training:
            self.down_sample(freq)
            self.add_features()
            self.add_target(short_mavg, long_mavg)
            self.clean()
            self.train_test_split()
            self.convert_to_floats()
            if transform == "standardize":
                self.standardize()
            else:
                self.scale()

            self.X_val.reset_index(drop=True, inplace=True)
            self.y_val.reset_index(drop=True, inplace=True)

        else:
            self.add_features()
            # convert to float type
            self.data = self.data.astype(float)
            self.clean()
            self.standardize()

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

        # add MA
        self.add_ma(n=10)
        self.add_ma(n=30)
        self.add_ma(n=200)
        # add OBV
        self.add_obv()

        # add Returns
        self.add_returns()
    # Calculation of moving average

    def add_returns(self):
        self.data["returns"] = np.log(self.data.Close / self.data.Close.shift())
    def add_ma(self, n):

        ma_str = 'MA_' + str(n)
        self.data[ma_str] = pd.Series(self.data['Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))

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
    def add_roc(self, n):
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

    def add_target(self, short_mavg=10, long_mavg=60):
        # Initialize the `signals` DataFrame with the `signal` column
        # datas['PriceMove'] = 0.0
        # Create short simple moving average over the short window
        self.data['short_mavg'] = self.data['Close'].rolling(window=short_mavg, min_periods=1, center=False).mean()
        # Create long simple moving average over the long window
        self.data['long_mavg'] = self.data['Close'].rolling(window=long_mavg, min_periods=1, center=False).mean()
        # Create signals
        self.data['signal'] = np.where(self.data['short_mavg'] > self.data['long_mavg'], 1, 0)

    def convert_to_floats(self):
        self.X_train = self.X_train.astype(float)
        self.X_val = self.X_val.astype(float)

    def clean(self):
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data.dropna(inplace=True)
        # self.close_prices = self.data.Close
        if self.training:
            self.data.drop(columns=["Volume", "Open", "High", "Low", "Close", "short_mavg", "long_mavg"], inplace=True)
        else:
            self.data.drop(columns=["Volume", "Open", "High", "Low", "Close"], inplace=True)

    def scale(self):
        scaler = MinMaxScaler()
        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.X_train.columns, index=self.X_train.index)
        self.X_val = pd.DataFrame(scaler.transform(self.X_val), columns=self.X_val.columns, index=self.X_val.index)

    def standardize(self):
        remainders = ["returns"]
        if self.training:
            self.features = [col for col in self.X_train.columns if col not in remainders]
        else:
            self.features = [col for col in self.data.columns if col not in remainders]

        ct = ColumnTransformer([
            ('scaler', StandardScaler(), self.features)
        ], remainder='passthrough')

        if self.training:
            self.features = [col for col in self.X_train.columns if col not in remainders]
            self.X_train = pd.DataFrame(ct.fit_transform(self.X_train),
                                      columns=self.features + remainders, index=self.X_train.index)
            self.X_val = pd.DataFrame(ct.transform(self.X_val),
                                      columns=self.features + remainders, index=self.X_val.index)
        else:
            #columns = list(self.data.columns)
            #index = self.data.index
            #self.data = ct.fit_transform(self.data.to_numpy())
            self.data = pd.DataFrame(ct.fit_transform(self.data),
                                      columns=self.data.columns, index=self.data.index)


    def train_test_split(self, split_idx=0.1, print_info=True):

        X = self.data.loc[:,  self.data.columns != "signal"]
        y = self.data.loc[:, "signal"]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y,
                                                                              test_size=split_idx,
                                                                              shuffle=False)
        if print_info:
            print("=" * 80)
            print(f"Shape of training set X_train: {self.X_train.shape}")
            print(f"Shape of training set y_train: {self.y_train.shape}")
            print(f"Shape of validation set X_val: {self.X_val.shape}")
            print(f"Shape of validation set y_val: {self.y_val.shape}")
            print("#" * 80)
            print(f"Training set start date: {self.X_train.index[0]}")
            print(f"Training set end date: {self.X_train.index[-1]}")
            print(f"Validation set start date: {self.X_val.index[0]}")
            print(f"Validation set end date: {self.X_val.index[-1]}")
            print("=" * 80)

    def perform_svd(self, ncomps=50, plot=True):

        self.svd = TruncatedSVD(n_components=ncomps)
        svd_fit = self.svd.fit(self.X_train_scaled_df)

        self.X_train_svd = self.svd.fit_transform(self.X_train_scaled_df)
        self.X_val_svd = self.svd.transform(self.X_val_scaled_df)

        self.X_train_svd_df = pd.DataFrame(self.X_train_svd,
                                           columns=['c{}'.format(c) for c in range(ncomps)],
                                           index=self.X_train_scaled_df.index)
        self.X_val_svd_df = pd.DataFrame(self.X_val_svd,
                                         columns=['c{}'.format(c) for c in range(ncomps)],
                                         index=self.X_val_scaled_df.index)

        if plot:
            plt_data = pd.DataFrame(svd_fit.explained_variance_ratio_.cumsum() * 100)
            plt_data.index = np.arange(1, len(plt_data) + 1)

            ax = plt_data.plot(kind='line', figsize=(10, 4))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Eigenvalues")
            ax.set_ylabel("Percentage Explained")
            ax.legend("")
            print('Variance preserved by first 50 components == {:.2%}'.format(
                svd_fit.explained_variance_ratio_.cumsum()[-1]))


if __name__ == "__main__":

    preparer = MlDataManager()
    symbol = "XRPUSDT"

    print(preparer.data.columns)
    print(preparer.data.shape)

