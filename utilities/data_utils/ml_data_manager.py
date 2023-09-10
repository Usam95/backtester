import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.decomposition import TruncatedSVD
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
        self.features = None

    def _get_path(self, symbol):
        symbol_folder = os.path.join(self.hist_data_folder, symbol)
        folder_contents = os.listdir(symbol_folder)
        file_formats = [f"{symbol}.parquet.gzip", f"{symbol}.csv"]

        for file_format in file_formats:
            if file_format in folder_contents:
                return os.path.join(symbol_folder, file_format)
            print(f"ERROR: Could not find any data for {symbol}..")
            return None

    def set_data(self, data):
        """
        Set data:
        :param Pandas DataFrame data: Historical data of cryptocurrency.
        """
        self.data = data.copy().astype(float)

    def load_data(self, symbol):
        data_path = self.get_path(symbol)
        if os.path.exists(data_path):
            _, file_extension = os.path.splitext(data_path)
            if file_extension == ".gzip":
                self.data = pd.read_parquet(data_path).astype(float)
            else:
                self.data = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date").astype(float)
            self.data.fillna(method="ffill", inplace=True)
        else:
            print(f"ERROR: The path {data_path} does not exist.")

    def down_sample(self, freq):
        freq_str = f"{freq}min"
        volume = self.data["Volume"].resample(freq_str).sum().iloc[:-1]
        self.data = self.data.loc[:, self.data.columns != "Volume"].resample(freq).last().iloc[:-1]
        self.data["Volume"] = volume

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

    def _scale(self):
        scaler = MinMaxScaler()
        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.X_train.columns, index=self.X_train.index)
        self.X_val = pd.DataFrame(scaler.transform(self.X_val), columns=self.X_val.columns, index=self.X_val.index)

    def _standardize(self):
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

