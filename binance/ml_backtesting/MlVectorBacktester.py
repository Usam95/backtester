from sklearn import linear_model
import pandas as pd 
import numpy as np

from MlDataManager import MlDataPreparer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from itertools import product
from MlDataPlotter import DataPlotML
import ta
import warnings
warnings.filterwarnings('ignore')


class MlVectorBacktester(MlDataPreparer):

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

    def __init__(self,  symbol, amount, tc=0.0007, model='logistic'):
        
        self.symbol = symbol
 
        self.amount = amount
        self.tc = tc
        self.results = None
        if model == 'regression':
            self.model = linear_model.LinearRegression()
        elif model == 'logistic':
            self.model = linear_model.LogisticRegression(C=1e5,
                solver='lbfgs', multi_class='ovr', max_iter=1000)

            self.model_name = 'LogisticRegression'
        else:
            raise ValueError('Model not known or not yet implemented.')

        self.get_data()

        self.dataploter = DataPlotML(self.symbol)

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        self.load_data(symbol=self.symbol)
        #self.data['returns'] = np.log(self.data.Close / self.data.Close.shift(1))
        #self.prepare_features_and_add_target(freq)
        #self.train_test_split()

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

    def train_test_split(self, split_idx=0.2, print_info=True):

        X_df = self.results.loc[:,  self.results.columns != "signal"]
        y_df = self.results.loc[:, "signal"]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_df, y_df,
                                                                              test_size=split_idx,
                                                                              shuffle=False)
        if print_info:
            print("=" * 80)
            print(f"Shape of training set X_train: {self.X_train.shape}")
            print(f"Shape of training set y_train: {self.y_train.shape}")
            print(f"Shape of training set X_val: {self.X_val.shape}")
            print(f"Shape of training set X_train: {self.y_val.shape}")
            print("#" * 80)
            print(f"Training set start date: {self.X_train.index[0]}")
            print(f"Training set end date: {self.X_train.index[-1]}")
            print(f"Validation set start date: {self.X_val.index[0]}")
            print(f"Validation set end date: {self.X_val.index[-1]}")
            print("=" * 80)

    def fit_model(self):

        ''' Implements the fitting step.
        '''
        self.model.fit(self.X_train[self.feature_columns], self.y_train)

    def run_strategy(self,  lags=3):
        ''' Backtests the trading strategy.
        '''
        self.lags = lags
        self.fit_model()

        self.y_pred = self.model.predict(self.X_val[self.feature_columns])
        self.X_val['prediction'] = self.y_pred
        self.X_val['strategy'] = (self.X_val['prediction']* self.X_val['returns'])
        # determine when a trade takes place
        trades = self.X_val['prediction'].diff().fillna(0) != 0
        # subtract transaction costs from return when trade takes place

        self.X_val['strategy'][trades] -= self.tc
        self.X_val['creturns'] = (self.amount * self.X_val['returns'].cumsum().apply(np.exp))
        self.X_val['cstrategy'] = (self.amount * self.X_val['strategy'].cumsum().apply(np.exp))

        self.results = self.X_val
        # absolute performance of the strategy
        aperf = self.results['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - self.results['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)

    def classification_report(self):
        # estimate accuracy on validation set
        print(accuracy_score(self.y_val, self.y_pred))
        print(confusion_matrix(self.y_val, self.y_pred))
        print(classification_report(self.y_val, self.y_pred))

    def confusion_matrix_elements(self):
        count = 0
        perf_metrics = ['TN', 'FP', 'FN', 'TP']
        self.perf_metrics_dic = {}
        for row in range((confusion_matrix(self.y_val, self.y_pred).shape[0])):
            for col in range((confusion_matrix(self.y_val, self.y_pred).shape[1])):
                metric_name = perf_metrics[count]
                metric_value = confusion_matrix(self.y_val, self.y_pred)[row][col]
                #print(f'{perf_metrics[count]}: {confusion_matrix(tester.y_val, tester.y_pred)[row][col]}')
                self.perf_metrics_dic[metric_name] = metric_value
                count += 1

        self.y_pred_train = self.model.predict(self.X_train[self.feature_columns])
        self.perf_metrics_dic["accuracy_val"] = round(accuracy_score(self.y_val, self.y_pred), 2)
        self.perf_metrics_dic["accuracy_train"] = round(accuracy_score(self.y_train, self.y_pred_train), 2)
        self.perf_metrics_dic["precision"] = round(precision_score(self.y_val, self.y_pred, average='binary'), 2)
        self.perf_metrics_dic["recall"] = round(recall_score(self.y_val, self.y_pred, average='binary'), 2)
        self.perf_metrics_dic["f1"] = round(f1_score(self.y_val, self.y_pred, average='binary'), 2)

    def runbacktest(self, freq):
        self.freq = freq
        self.prepare_features_and_add_target(freq, lags=5)
        self.train_test_split()
        self.fit_model()

        self.y_pred = self.model.predict(self.X_val[self.feature_columns])
        self.confusion_matrix_elements()

        self.dataploter.store_testcase_data(self)

    def optimize_strategy(self, freq_range):  # Adj!!!
        freqs = range(*freq_range)
        performance = []
        for freq in freqs:
            self.runbacktest(freq)

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = '%s | TC = %.4f' % (self.symbol, self.tc)
        self.results[['creturns', 'cstrategy']].plot(title=title,
                                                     figsize=(10, 6))


if __name__ == "__main__":
    symbol = "XRPUSDT"
    tester = ScikitVectorBacktester(symbol=symbol, amount=1000)
    tester.optimize_strategy((30, 1200, 30))
    tester.dataploter.df.to_csv("ml_backtest_results2.csv", index=False)