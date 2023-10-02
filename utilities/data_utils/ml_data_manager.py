import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import TruncatedSVD
from matplotlib.ticker import MaxNLocator

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .data_loader import load_data

from .ml_feature_engineer import FeatureEngineer


# Set display options
pd.set_option('display.max_columns', None)  # None means show all columns
pd.set_option('display.width', None)        # Ensure full width is utilized
pd.set_option('display.max_colwidth', None)   # Ensure each column is wide enough to show full content


class MlDataManager:
    def __init__(self, config,  hist_data_folder="../../hist_data", training=True):
        self.config = config
        self.hist_data_folder = hist_data_folder
        self.symbol = config.symbol
        self.training = training
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.features = None
        self.feature_engineer = None
        self.load_data()
        self.periods = [10, 30, 50, 70]

    def load_data(self):
        self.data = load_data(self.symbol)
        self._validate_data()

    def set_data(self, data):
        """
        Set data:
        :param Pandas DataFrame data: Historical data of cryptocurrency.
        """
        self.data = data.copy().astype(float)

    def _validate_data(self):
        """Validate the input data for necessary columns."""
        required_columns = ['Close', 'Volume', 'High', 'Low']

        missing_columns = [col for col in required_columns if col not in self.data.columns]

        if missing_columns:
            raise ValueError(f"Input data is missing the following columns: {', '.join(missing_columns)}")

    def resample_data(self):
        volume = self.data["Volume"].resample(self.config.freq).sum().iloc[:-1]
        self.data = self.data.loc[:, self.data.columns != "Volume"].resample(self.config.freq).last().iloc[:-1]
        self.data["Volume"] = volume

    def transform_data(self):
        """
        Transform the dataset by applying various transformations.

        Sets:
        - self.results (pd.DataFrame): Transformed dataset.
        """

        # Log transformation for skewed distributions. Add "+ 1e-10" to avoid log(0)
        self.X_train['Volume'] = np.log(self.X_train['Volume'] + 1e-10)
        #self.X_train['OBV'] = np.log1p(self.X_train['OBV'] + 1e-10)

        self.X_test['Volume'] = np.log(self.X_test['Volume'] + 1e-10)
        #self.X_test['OBV'] = np.log1p(self.X_test['OBV'] + 1e-10)

        # StandardScaler for price columns
        price_cols = ['Open', 'High', 'Low', 'Close']
        scaler_price = StandardScaler()
        self.X_train[price_cols] = scaler_price.fit_transform(self.X_train[price_cols])
        self.X_test[price_cols] = scaler_price.transform(self.X_test[price_cols])

        # Dynamically construct the list of indicators using self.periods
        indicators = []
        for n in self.periods:
            indicators.extend([
                f"MA_{n}", f"EMA_{n}", f"Momentum_{n}", f"ROC_{n}", f"STOCK_{n}", f"STOD_{n}",
                f"STOS_{n}", f"STOMOM_{n}", f"STOROC_{n}", f"StochRSI_{n}", f"STOCROSS_{n}"
            ])


        scaler_indicators = StandardScaler()
        self.X_train[indicators] = scaler_indicators.fit_transform(self.X_train[indicators])
        self.X_test[indicators] = scaler_indicators.transform(self.X_test[indicators])

        # StandardScaler for RSI
        rsi_cols = [f"RSI_{n}" for n in self.periods]
        scaler_rsi = StandardScaler()
        self.X_train[rsi_cols] = scaler_rsi.fit_transform(self.X_train[rsi_cols])
        self.X_test[rsi_cols] = scaler_rsi.transform(self.X_test[rsi_cols])

    def print_data_info(self):
        """
        Print the shape and date range for the training and validation datasets.
        """

        print("=" * 80)
        print(f"Shape of training set X_train: {self.X_train.shape}")
        print(f"Shape of training set y_train: {self.y_train.shape}")
        print(f"Shape of validation set X_val: {self.X_test.shape}")
        print(f"Shape of validation set y_val: {self.y_test.shape}")
        print("#" * 80)
        print(f"Training set start date: {self.X_train.index[0]}")
        print(f"Training set end date: {self.X_train.index[-1]}")
        print(f"Validation set start date: {self.X_test.index[0]}")
        print(f"Validation set end date: {self.X_test.index[-1]}")
        print("=" * 80)
        print(f"Type of training set X_train: {type(self.X_train)}")
        print(f"Type of training set y_train: {type(self.y_train)}")
        print(f"Type of validation set X_val: {type(self.X_test)}")
        print(f"Type of validation set y_val: {type(self.y_test)}")
        print("=" * 80)
        train_trades = self.y_train.diff().abs().shift(1).value_counts().get(1.0, 0)
        test_trades = self.y_test.diff().abs().shift(1).value_counts().get(1.0, 0)
        print(f"Num of trades in training set: {train_trades}")
        print(f"Num of trades in testing set: {test_trades}")
        print("=" * 80)

    def extract_feature_columns(self):
        excluded_columns = ['Signal'] #'Open', 'High', 'Low',
        self.feature_columns = [col for col in self.data.columns if col not in excluded_columns]

    def _split_data(self, config, test_size=0.2):
        """
        Split the dataset into train and test sets.

        Args:
        - config (dict): Configuration settings including split_date.
        - test_size (float): Proportion of the dataset to include in the test set if no split_date is provided.

        Sets:
        - self.X_train (pd.DataFrame): Training data without target variable.
        - self.X_test (pd.DataFrame): Test data without target variable.
        - self.y_train (pd.Series): Target variable for training data.
        - self.y_test (pd.Series): Target variable for test data.
        """

        # Extract the target variable 'Signal' for both training and testing datasets
        y = self.data['Signal']

        split_date = config["dataset_conf"]["split_date"]

        if split_date != "":  # If split_date is provided in the config
            mask = self.data.index <= split_date
            X_train, X_test = self.data[mask], self.data[~mask]
            y_train, y_test = y[mask], y[~mask]
        else:  # If no split_date is provided, use the test_size to split the data
            X_train, X_test, y_train, y_test = train_test_split(self.data, y, test_size=test_size, shuffle=False)

        # Drop the 'Signal' column from the training and testing data
        X_train = X_train.drop(columns=['Signal'])
        X_test = X_test.drop(columns=['Signal'])

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def preprocess_data(self):
        self.resample_data()
        self.feature_engineer = FeatureEngineer(self.data, self.periods)
        self.data = self.feature_engineer.add_features()
        self.data = self.feature_engineer.add_target(self.config)
        self.extract_feature_columns()
        self._clean_data()
        self._split_data()
        self.transform_data()
        self.print_data_info()
        self.perform_svd()

    def _convert_to_floats(self):
        self.X_train = self.X_train.astype(float)
        self.X_val = self.X_val.astype(float)

    def _clean_data(self):
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.dropna(inplace=True)
        # Exclude the last row
        self.data = self.data[:-1]

    def perform_svd(self, ncomps=30):
        """
        Perform Singular Value Decomposition (SVD) on the training data.

        :param ncomps: Number of components for SVD.
        """
        # Instantiate and fit the SVD object
        self.svd = TruncatedSVD(n_components=ncomps)
        self.svd.fit(self.X_train)

        # Transform the data sets
        self.X_train_svd = self.svd.transform(self.X_train)
        self.X_test_svd = self.svd.transform(self.X_test)

        # Convert the transformed data to DataFrames with appropriate column names
        columns = [f'c{i}' for i in range(ncomps)]
        self.X_train_svd_df = pd.DataFrame(self.X_train_svd, columns=columns, index=self.X_train.index)
        self.X_test_svd_df = pd.DataFrame(self.X_test_svd, columns=columns, index=self.X_test.index)


if __name__ == "__main__":

    symbol = "XRPUSDT"
    preparer = MlDataManager(symbol=symbol)
    preparer.preprocess_data()

    print(preparer.data.columns)
    print(preparer.data.shape)
    print("="*100)
    print("\n")
    print(preparer.results.columns)
    print(preparer.results.shape)
    preparer.results.dropna(inplace=True)
    print(preparer.results.head(5))

    preparer.transform_data()
    print("="*100)
    print("\n")
    print(preparer.results.head(5))


