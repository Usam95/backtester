
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

base_str_data = ['symbol', 'model_name', 'accuracy_train', 'accuracy_val', 'precision', 'recall',
                 'f1', 'TN', 'FP', 'FN', 'TP', 'freq', 'train_samples', 'val_samples', "short_mavg",
                 "long_mavg", "aperf", "operf", "num_trades"
                 ]


class MlDataPlotter:

    def __init__(self):
        self.df = None
        self.data_manager = None
        self.init_df()

    def init_df(self):
        self.df = pd.DataFrame(columns=base_str_data)

    def set_data_manager(self, data_manager):
        self.data_manager = data_manager

    def store_testcase_data(self, model_perf_data):

        testcase_data = []
        testcase_data.append(model_perf_data["symbol"])
        testcase_data.append(model_perf_data["model_name"])
        testcase_data.append(model_perf_data['accuracy_train'])
        testcase_data.append(model_perf_data['accuracy_val'])
        testcase_data.append(model_perf_data['precision'])
        testcase_data.append(model_perf_data['recall'])
        testcase_data.append(model_perf_data['f1'])
        testcase_data.append(model_perf_data['TN'])
        testcase_data.append(model_perf_data['FP'])
        testcase_data.append(model_perf_data['FN'])
        testcase_data.append(model_perf_data['TP'])
        testcase_data.append(model_perf_data["freq"])
        testcase_data.append(model_perf_data["train_samples"])
        testcase_data.append(model_perf_data["val_samples"])
        testcase_data.append(model_perf_data["short_mavg"])
        testcase_data.append(model_perf_data["long_mavg"])
        testcase_data.append(model_perf_data["aperf"])
        testcase_data.append(model_perf_data["operf"])
        testcase_data.append(model_perf_data["num_trades"])

        self.df.loc[len(self.df)] = testcase_data

    def plotModelFeatureImportance(self, model):
        Importance = pd.DataFrame({'Importance': model.feature_importances_ * 100}, index=self.X_train_svd_df.columns)
        Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r')
        plt.xlabel('Variable Importance')

    def plotDataSetHistogram(self):
        # histograms
        self.data.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(18, 30))
        plt.show()

    def plotSignalDistribution(self, ):
        fig = plt.figure()
        plot = self.y_train.to_frame('signal').groupby(['signal']).size().plot(kind='barh', color='red')
        plt.title("Signal values distribution of training dataset.")
        plt.show()

        fig = plt.figure()
        plot = self.y_val.to_frame('signal').groupby(['signal']).size().plot(kind='barh', color='red')
        plt.title("Signal values distribution of validation dataset.")
        plt.show()

    def plotCovarianceMatrix(self):
        correlation = self.data.corr()
        plt.figure(figsize=(30, 30))
        plt.title('Correlation Matrix')
        sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')

    def plotSignalsWithTSNE(self):
        svdcols = [c for c in self.X_train_svd_df.columns if c[0] == 'c']

        tsne = TSNE(n_components=2, random_state=0)
        Z = tsne.fit_transform(self.X_train_svd_df[svdcols])

        dftsne = pd.DataFrame(Z, columns=['x', 'y'], index=self.X_train_svd_df.index)
        dftsne['signal'] = self.y_train

        sns.set(rc={"figure.figsize": (16, 8)})  # width=8, height=4
        g = sns.lmplot(data=dftsne, x='x', y='y', hue='signal', fit_reg=False,
                       scatter_kws={'alpha': 0.4, 's': 60}, height=8, aspect=1.5)

    def plot_confusion_matrix(self, data_manager):
        print(confusion_matrix(data_manager.y_val, data_manager.y_pred))
        df_cm = pd.DataFrame(confusion_matrix(data_manager.y_val, data_manager.y_pred), columns=np.unique(data_manager.y_val),
                             index=np.unique(data_manager.y_val))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})  # font sizes

    def classification_report(self):
        # estimate accuracy on validation set
        print(accuracy_score(self.y_val, self.y_pred))
        print(confusion_matrix(self.y_val, self.y_pred))
        print(classification_report(self.y_val, self.y_pred))




    def plot_model_evaluation_res(self, results, names):
        # compare algorithms
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        fig.set_size_inches(15, 8)
        plt.show()

    def plot_backtest_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = '%s | TC = %.4f' % (self.symbol, self.tc)
        self.results[['creturns', 'cstrategy']].plot(title=title,
                                                     figsize=(10, 6))
