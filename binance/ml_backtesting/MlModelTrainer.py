# Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#Libraries for Deep Learning Models
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
#from keras.optimizers import SGD

from statsmodels.tsa.stattools import adfuller

from MlDataManager import MlDataManager

from scipy.linalg import svd   
from scipy import stats
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MaxNLocator
import ta


class BlockingTimeSeriesSplit:
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self):
        return self.n_splits
    
    def split(self, X):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]
            

class MlModelTrainer:
   
    def __init__(self):
        self.models = []
        self.model = None
        self.init_models()

    def init_models(self):
        # spot check the algorithms
        self.models.append(('LR', LogisticRegression(n_jobs=-1)))
        self.models.append(('LDA', LinearDiscriminantAnalysis()))
        self.models.append(('KNN', KNeighborsClassifier()))
        self.models.append(('CART', DecisionTreeClassifier()))
        self.models.append(('NB', GaussianNB()))
        #Neural Network
        self.models.append(('NN', MLPClassifier()))
        #Ensable Models 
        # Boosting methods
        self.models.append(('AB', AdaBoostClassifier()))
        self.models.append(('GBM', GradientBoostingClassifier()))
        # Bagging methods
        self.models.append(('RF', RandomForestClassifier(n_jobs=-1)))
        
    def train_models(self, data_manager, svd=False, scoring='accuracy', num_folds=2):
        # create list of models
        # test options for classification
        scoring = 'accuracy'
        #scoring = 'precision'
        #scoring = 'recall'
        #scoring ='neg_log_loss'
        #scoring = 'roc_auc'
        seed = 42
        self.results = []
        self.names = []
        
        for name, model in self.models:
            kfold = KFold(n_splits=num_folds)
            #btscv = BlockingTimeSeriesSplit(n_splits=num_folds)
            if svd: 
                cv_results = cross_val_score(model, self.X_train_svd_df, self.y_train, cv=kfold, scoring=scoring)
            else:
                cv_results = cross_val_score(model, data_manager.X_train, data_manager.y_train,
                                             cv=kfold, scoring=scoring)

            self.results.append(cv_results)
            self.names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg) 
        
    def optimize_model_parameters(self):
        parameters = {
                    'penalty': ['l1', 'l2'],
                    'C': np.logspace(-3,3,7),
                    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
                    }

        reg = LogisticRegression()
        grid = GridSearchCV(estimator=reg, param_grid = parameters, cv = 1, n_jobs=-1)
        grid.fit(self.X_train_svd_df, self.y_train)

    def train_evaluate_model(self, data_manager, model_perf_data, scaled=True):
        self.model = LogisticRegression()
        model_name = "LR"
        model_perf_data["model_name"] = model_name
        if scaled:
            self.model.fit(data_manager.X_train, data_manager.y_train)
            data_manager.y_pred = self.model.predict(data_manager.X_val)
            # calculate the performance metrics of trained model
            self.calculate_confusion_matrix(data_manager, model_perf_data)
            # store the calculated performance metrics
            #data_plotter.store_testcase_data(perf_metrics_dic)

            # print("Accuracy:", self.model.score(data_manager.X_val, data_manager.y_val))

        else:
            self.model.fit(self.X_train, self.y_train)
            self.y_pred = self.model.predict(self.X_val)
            # print("Accuracy:",self.model.score(self.X_val, self.y_val)

    def calculate_confusion_matrix(self, data_manager, perf_metrics_dic):
        count = 0
        perf_metrics = ['TN', 'FP', 'FN', 'TP']
        for row in range((confusion_matrix(data_manager.y_val, data_manager.y_pred).shape[0])):
            for col in range((confusion_matrix(data_manager.y_val, data_manager.y_pred).shape[1])):
                metric_name = perf_metrics[count]
                metric_value = confusion_matrix(data_manager.y_val, data_manager.y_pred)[row][col]
                # print(f'{perf_metrics[count]}: {confusion_matrix(tester.y_val, tester.y_pred)[row][col]}')
                perf_metrics_dic[metric_name] = metric_value
                count += 1

        # compute predictions on validation data
        data_manager.y_pred = self.model.predict(data_manager.X_val)
        # compute predictions on training data
        y_pred_train = self.model.predict(data_manager.X_train)
        perf_metrics_dic["accuracy_val"] = round(accuracy_score(data_manager.y_val, data_manager.y_pred), 2)
        perf_metrics_dic["accuracy_train"] = round(accuracy_score(data_manager.y_train, y_pred_train), 2)
        perf_metrics_dic["precision"] = round(precision_score(data_manager.y_val, data_manager.y_pred, average='binary'), 2)
        perf_metrics_dic["recall"] = round(recall_score(data_manager.y_val, data_manager.y_pred, average='binary'), 2)
        perf_metrics_dic["f1"] = round(f1_score(data_manager.y_val, data_manager.y_pred, average='binary'), 2)

        return perf_metrics_dic
