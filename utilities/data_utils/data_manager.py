import pandas as pd
import os

from ta.momentum import RSIIndicator

class ParameterLoader:

    def __init__(self, symbol=None):
        self.hist_data_folder = "../results"
        self.EMA_folder = "EMA"
        self.BB_folder = "BB"
        self.indicator_params = {}

        if symbol is not None:
            self.symbol = symbol

    def set_symbol(self, symbol):
        self.symbol = symbol

    def get_ema_params(self):

        df_path = os.path.join(self.hist_data_folder, self.symbol, self.EMA_folder, "backtest_test_res.csv")
        df = pd.read_csv(df_path)
        ema_s = df.nlargest(1, 'outperf').reset_index(drop=True).loc[0, "EMA_S"]
        ema_l = df.nlargest(1, 'outperf').reset_index(drop=True).loc[0, "EMAL"]
        freq = df.nlargest(1, 'outperf').reset_index(drop=True).loc[0, "freq"]
        print(f"EMA best frequency: {freq}")
        self.indicator_params["EMA"] = {"EMA_S":ema_s, "EMA_L":ema_l}

    def get_bb_params(self):

        df_path = os.path.join(self.hist_data_folder, self.symbol, self.BB_folder, "backtest_test_res.csv")
        df = pd.read_csv(df_path)
        sma = df.nlargest(1, 'outperf').reset_index(drop=True).loc[0, "SMA"]
        dev = df.nlargest(1, 'outperf').reset_index(drop=True).loc[0, "Dev"]

        freq = df.nlargest(1, 'outperf').reset_index(drop=True).loc[0, "freq"]
        print(f"BB best frequency: {freq}")
        self.indicator_params["BB"] = {"SMA":sma, "Dev":dev}


    def load_indicator_params(self):
        self.get_ema_params()
        self.get_bb_params()


class DataManager(ParameterLoader):

    def __init__(self, symbol):

        self.hist_data_folder = "../hist_data"
        if symbol is not None:
            self.symbol = symbol
            self.params_loader = ParameterLoader(symbol=symbol)
            self.params_loader.load_indicator_params()


    def _get_path(self):
        symbol_folder = os.path.join(self.hist_data_folder, self.symbol)
        folder_contents = os.listdir(symbol_folder)
        if f"{self.symbol}.parquet.gzip" in folder_contents:
            data_path = os.path.join(symbol_folder, f"{self.symbol}.parquet.gzip")
            return data_path
        elif f"{self.symbol}.csv" in folder_contents:
            data_path = os.path.join(symbol_folder, f"{self.symbol}.csv")
            return data_path
        else:
            print(f"ERROR: Could not find any data for {self.symbol}..")
            exit(0)

    def load_data(self):
        data_path = self._get_path()
        _, file_extension = os.path.splitext(data_path)
        if file_extension == ".gzip":
            self.data = pd.read_parquet(data_path)
        else:
            self.data = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")


    def set_symbol(self, symbol):
        self.symbol = symbol

    def _resample(self):

        freq = "{}min".format(40)
        self.results = self.results.resample(freq).last().dropna().iloc[:-1]

    def prepare_data(self):
        self.results = self.data.copy()
        self._resample()
        self._calculate_returns()
        self._calculate_ema()
        self._calculate_bb()
        self._calculate_range()
        self._calculate_rsi()
        self._add_day_of_week()
        self._add_time_intervals()
        self._add_rolling_cum_range()
        self._add_rolling_cum_returns()

        # feature engineering
        self._scale_features()
        self.results.dropna(inplace=True)
        # calculate EMA

    def _calculate_returns(self):
        # calculate returns
        self.results["Returns"] = self.results.Close.pct_change(1)

    def _calculate_range(self):
        self.results["Range"] = self.results["High"] / self.results["Low"] - 1

    def _calculate_ema(self):
        ema_s = self.params_loader.indicator_params["EMA"]["EMA_S"]
        ema_l = self.params_loader.indicator_params["EMA"]["EMA_L"]
        self.results["EMA_S"] = self.results["Close"].ewm(span=ema_s, min_periods=ema_s).mean()
        self.results["EMA_L"] = self.results["Close"].ewm(span=ema_l, min_periods=ema_l).mean()

    def _calculate_bb(self):
        bb_sma = self.params_loader.indicator_params["BB"]["SMA"]
        bb_dev = self.params_loader.indicator_params["BB"]["Dev"]
        self.results["BB_SMA"] = self.results["Close"].rolling(window=bb_sma).mean()
        self.results["BB_Dev"] = self.results["Close"].rolling(window=bb_sma).std() * bb_dev

    def _calculate_rsi(self):
        rsi = RSIIndicator(close=self.results["Close"], window=14).rsi()
        self.results["RSI"] = rsi
        self.results["RSI_Ret"] = self.results["RSI"] / self.results["RSI"].shift(1)

    def _add_day_of_week(self):
        self.results["DOW"] =  self.results.index.dayofweek

    def _add_time_intervals(self):
        t_steps = [1, 2]
        t_features = ["Returns", "Range", "RSI_Ret"]
        for ts in t_steps:
            for tf in t_features:
                self.results[f"{tf}_T{ts}"] = self.results[tf].shift(ts)

    def _scale_features(self):
        self.results[["Open", "High", "Low", "Volume"]] = self.results[["Open", "High", "Low", "Volume"]].pct_change()

    def _add_rolling_cum_returns(self):
        self.results["Roll_Rets"] = self.results["Returns"].rolling(window=30).sum()

    def _add_rolling_cum_range(self):
        self.results["Avg_Range"] = self.results["Range"].rolling(window=30).mean()

if __name__ == "__main__":
    symbol = "VGXUSDT"
    manager = DataManager(symbol)
    manager.load_data()
    manager.prepare_data()

    print(manager.results.columns)
    manager.results.to_csv(f"../strategies/{symbol}_preprocessed.csv")
   #  rint(manager.params_loader.indicator_params)
    # paramLoader = ParameterLoader()
    # paramLoader.set_symbol("VGXUSDT")
    #
    # paramLoader.get_ema_params()
    # paramLoader.get_bb_params()
    #
    # print(paramLoader.indicator_params)

