from sklearn import linear_model
import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from itertools import product
from MlDataPlotter import MlDataPlotter
from MlDataManager import MlDataManager
from MlModelTrainer import MlModelTrainer

from itertools import product

import ta
import warnings
warnings.filterwarnings('ignore')


class MlVectorBacktester:

    ''' Class for the vectorized backtesting of
    Machine Learning-based trading strategies.
    Attributes
    ==========
    symbol: str
        TR RIC (financial instrument) to work with
    start: str
        start date for data selection
    end: str
        end date for data selection
    amount: int, float
        amount to be invested at the beginning
    tc: float
        proportional transaction costs (e.g. 0.5% = 0.005) per trade
    model: str
        either 'regression' or 'logistic'
    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    select_data:
        selects a sub-set of the data
    prepare_features:
        prepares the features data for the model fitting
    fit_model:
        implements the fitting step
    run_strategy:
        runs the backtest for the regression-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol
    '''

    def __init__(self, symbol, amount=1, tc=0.0007):
        
        self.symbol = symbol
 
        self.amount = amount
        self.tc = tc
        self.results = None
        # dictionary to store model performance data
        self.model_perf_data = {}
        self.data_plotter = MlDataPlotter()
        self.data_manager = MlDataManager()
        self.model_trainer = MlModelTrainer()

    def prepare_data_and_model(self, freq, short_mavg, long_mavg):
        ''' Retrieves and prepares the data.
        '''
        self.data_manager.load_data(symbol=self.symbol)
        self.data_manager.preprocess_data(freq=freq, short_mavg=short_mavg, long_mavg=long_mavg)

        self.model_trainer.train_evaluate_model(self.data_manager, self.model_perf_data)

        # store used cryptocurrency ticker
        self.model_perf_data["symbol"] = self.symbol

    def upsample(self):
        '''  Upsamples/copies trading positions back to higher frequency.
        '''

        data = self.data.copy()
        resamp = self.results.copy()

        data["position"] = resamp.position.shift()
        data = data.loc[resamp.index[0]:].copy()
        data.position = data.position.shift(-1).ffill()
        data.dropna(inplace=True)
        self.results = data

    def run_strategy(self):
        ''' Backtests the trading strategy.
        '''
        self.data_manager.X_val['strategy'] = pd.Series(self.data_manager.y_pred).shift() * self.data_manager.X_val['returns']
        # determine when a trade takes place
        self.data_manager.X_val["trades"] = pd.Series(self.data_manager.y_pred).diff().fillna(0).abs()
        self.data_manager.X_val["trades"].fillna(0, inplace=True)
        # subtract transaction costs from return when trade takes place
        #print(f"3: {self.data_manager.X_val.isnull().values.any()}")

        self.data_manager.X_val.strategy = self.data_manager.X_val.strategy - self.data_manager.X_val["trades"] * self.tc
        # self.data_manager.X_val.strategy = self.data_manager.X_val.strategy - self.tc
        # print(f"4: {self.data_manager.X_val.isnull().values.any()}")
        # self.data_manager.X_val['strategy'] = self.data_manager.X_val['strategy'] - self.tc
        self.data_manager.X_val['creturns'] = self.data_manager.X_val['returns'].cumsum().apply(np.exp)
        self.data_manager.X_val['cstrategy'] = self.data_manager.X_val['strategy'].cumsum().apply(np.exp)

        # self.results = self.X_val
        # absolute performance of the strategy
        aperf = self.data_manager.X_val['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - self.data_manager.X_val['creturns'].iloc[-1]
        # number of trades
        num_trades = self.data_manager.X_val['trades'].sum()
        return round(aperf, 2), round(operf, 2), num_trades

    def classification_report(self):
        # estimate accuracy on validation set
        print(accuracy_score(self.y_val, self.y_pred))
        print(confusion_matrix(self.y_val, self.y_pred))
        print(classification_report(self.y_val, self.y_pred))

    def runbacktest(self, freq, short_mavg, long_mavg):
        # load and prepare the data
        self.prepare_data_and_model(freq, short_mavg, long_mavg)
        aperf, operf, num_trades = self.run_strategy()

        # store testcase parameters
        self.model_perf_data["freq"] = freq
        self.model_perf_data["short_mavg"] = short_mavg
        self.model_perf_data["long_mavg"] = long_mavg
        self.model_perf_data["train_samples"] = len(self.data_manager.X_train)
        self.model_perf_data["val_samples"] = len(self.data_manager.X_val)
        self.model_perf_data["aperf"] = aperf
        self.model_perf_data["operf"] = operf
        self.model_perf_data["num_trades"] = num_trades

        self.data_plotter.store_testcase_data(self.model_perf_data)
        print(f"freq: {freq} | aperf: {aperf} | outperf: {operf}")

    def optimize_strategy(self, freq_range, short_mavg_range, long_mavg_range):  # Adj!!!
        freqs = range(*freq_range)
        short_mavgs = range(*short_mavg_range)
        long_mavgs = range(*long_mavg_range)
        combinations_ = list(product(freqs, short_mavgs, long_mavgs))
        combinations = [comb for comb in combinations_ if comb[1] < comb[2]]
        print(len(combinations))
        for comb in combinations:
            self.runbacktest(comb[0], comb[1], comb[2])


if __name__ == "__main__":
    symbol = "XRPUSDT"
    backtester = MlVectorBacktester(symbol=symbol)
    freq_range = (15, 600, 15)
    short_mavg_range = (10, 120, 15)
    long_mavg_range = (45, 200, 15)
    backtester.optimize_strategy(freq_range, short_mavg_range, long_mavg_range)
    backtester.data_plotter.df.to_csv("ml_backtest_res.csv")
    # Adj!!!