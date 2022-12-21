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
   
    def __init__(self, symbol):
        self.symbol = symbol
        self.data_manager = MlDataManager()
        self.prepare_data()
        
    def prepare_data(self):
        self.model_trainer.load_data(self.symbol)

    def add_ta(self): 
        df = self.data.copy()
        df_indicators = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", 
                                              fillna=True).shift(1)
        self.data = pd.concat((self.data, df))

    def add_range_target(self):
        # Specify Target
        self.data["signal"] = 0
        self.data.loc[self.data["range"].shift(-1) >  self.data["avg_range"], "signal"] = 1
        self.data.loc[self.data["range"].shift(-1) <=  self.data["avg_range"], "signal"] = 0    

    def add_close_target(self):
        self.data['priceDirection'] = (self.data['Close'].shift(-1) -  self.data['Close'])
        self.data['signal'] = 0.0
        self.data['signal'] = np.where((self.data.loc[:,'priceDirection'] > 0), 1.0, 0.0)
        self.data.drop(columns=['priceDirection'], inplace=True)
        
        
    def preprocess(self, target="close"):
        #df = self.copy()  
        self.add_ta()
        self.data['closeReturn'] = self.data['Close'].pct_change()
        self.data['highReturn'] = self.data['High'].pct_change()
        self.data['lowReturn'] = self.data['Low'].pct_change()
        self.data['volReturn'] = self.data['Volume'].pct_change()
        self.data['dailyChange'] = (self.data['Close'] - self.data['Open']) / self.data['Open']

        if target == "close":
            self.add_close_target()
        else:
            self.add_range_target()

        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data.drop(columns =['Open', 'High', 'Low', 'Close', "dir", 'Volume', ], axis=1, inplace=True)
        self.data.dropna(inplace=True)

    def train_test_split(self, split_idx=0.2, print_info=True):
        
        self.data.dropna(how='any', inplace=True)
        X_df = self.data.loc[:, self.data.columns!="signal"]
        y_df = self.data.loc[:,"signal"]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_df, y_df, 
                                                                              test_size=split_idx,
                                                                              shuffle=False)
        if print_info: 
            print("="*80)
            print(f"Shape of training set X_train: {self.X_train.shape}")
            print(f"Shape of training set y_train: {self.y_train.shape}")
            print(f"Shape of training set X_val: {self.X_val.shape}")
            print(f"Shape of training set y_train: {self.y_val.shape}")
            print("#"*80)
            print(f"Training set start date: {self.X_train.index[0]}")
            print(f"Training set end date: {self.X_train.index[-1]}")
            print(f"Validation set start date: {self.X_val.index[0]}")
            print(f"Validation set end date: {self.X_val.index[-1]}") 
            print("="*80)
            
    def scale(self, ncomps=60):
        scaler = StandardScaler().fit(self.X_train)
        self.X_train_scaled_df = pd.DataFrame(scaler.transform(self.X_train), 
                                              columns = self.X_train.columns, 
                                              index = self.X_train.index)

        self.X_val_scaled_df = pd.DataFrame(scaler.transform(self.X_val), 
                                              columns = self.X_val.columns, 
                                              index = self.X_val.index)


        self.X_train.dropna(how='any', inplace=True)
        self.X_train_scaled_df.dropna(how='any', inplace=True)
        
        self.X_val.dropna(how='any', inplace=True)
        self.X_val_scaled_df.dropna(how='any', inplace=True)
        
    def perform_SVD(self, ncomps=50,plot=True):

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
            print('Variance preserved by first 50 components == {:.2%}'.format(svd_fit.explained_variance_ratio_.cumsum()[-1]))
    
    def plotModelFeatureImportance(self, model):

        Importance = pd.DataFrame({'Importance':model.feature_importances_*100}, index=self.X_train_svd_df.columns)
        Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r' )
        plt.xlabel('Variable Importance')
       
    def plotDataSetHistogram(self):
        
        # histograms
        self.data.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1, figsize=(18,30))
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
        plt.figure(figsize=(30,30))
        plt.title('Correlation Matrix')
        sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')

    def plotSignalsWithTSNE(self):
        svdcols = [c for c in self.X_train_svd_df.columns if c[0] == 'c']

        tsne = TSNE(n_components=2, random_state=0)
        Z = tsne.fit_transform(self.X_train_svd_df[svdcols])

        dftsne = pd.DataFrame(Z, columns=['x','y'], index=self.X_train_svd_df.index)
        dftsne['signal'] = self.y_train

        sns.set(rc={"figure.figsize":(16, 8)}) #width=8, height=4
        g = sns.lmplot(data=dftsne, x='x', y='y', hue='signal', fit_reg=False, 
                       scatter_kws={'alpha':0.4,'s':60}, height=8, aspect=1.5)
        
    def init_models(self):
        # spot check the algorithms
        self.models = []
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
        
    def train_models(self, svd=True, scoring='accuracy', num_folds=6):
        # create list of models
        self.init_models()
        
        # test options for classification
        scoring = 'accuracy'
        #scoring = 'precision'
        #scoring = 'recall'
        #scoring ='neg_log_loss'
        #scoring = 'roc_auc'
        seed = 7
        results = []
        names = []
        
        for name, model in self.models:
            kfold = KFold(n_splits=num_folds)
            btscv = BlockingTimeSeriesSplit(n_splits=num_folds)

            if svd: 
                cv_results = cross_val_score(model, self.X_train_svd_df, self.y_train, cv=btscv, scoring=scoring)
            else:
                cv_results = cross_val_score(model, self.X_train, self.y_train, cv=btscv, scoring=scoring)

            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg) 
        
    def optimize_model_parameters(self):
        parameters = {
                    'penalty' : ['l1','l2'], 
                    'C'       : np.logspace(-3,3,7),
                    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
                    }

        reg = LogisticRegression()

        grid = GridSearchCV(estimator=reg, param_grid = parameters, cv = 1, n_jobs=-1)
        grid.fit(self.X_train_svd_df, self.y_train)
        
    
    def train_evaluate_model(self, scaled=True):
        reg = LogisticRegression(C = 0.01, 
                            penalty = 'l1', 
                            solver = 'liblinear')
        
        if scaled: 
            reg.fit(self.X_train_svd_df, self.y_train)
            self.y_pred = reg.predict(self.X_val_svd)
            print("Accuracy:",reg.score(self.X_val_svd, self.y_val))
    
        else: 
            reg.fit(self.X_train, self.y_train)
            self.y_pred = reg.predict(self.X_val)
            print("Accuracy:",reg.score(self.X_val, self.y_val))

    def plot_confusion_matrix(self):
        
        print(confusion_matrix(self.y_val, self.y_pred))
        df_cm = pd.DataFrame(confusion_matrix(self.y_val, self.y_pred), columns=np.unique(self.y_val), index=np.unique(self.y_val))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size":16}) #font sizes

    
    def classification_report(self):
        # estimate accuracy on validation set
        print(accuracy_score(self.y_val, self.y_pred))
        print(confusion_matrix(self.y_val, self.y_pred))
        print(classification_report(self.y_val, self.y_pred))
