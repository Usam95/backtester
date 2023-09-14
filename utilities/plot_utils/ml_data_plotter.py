
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

# set seaborn style
sns.set_style("whitegrid")
sns.set_context("talk")  # this will make plot elements slightly larger

base_ml_data = [
    'symbol', 'model_name', 'acc_train', 'acc_val', 'precision',
    'recall', 'f1', 'TN', 'FP', 'FN', 'TP', 'freq', 'len_train',
    'len_val',  'aperf', 'operf', 'num_trades'
]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12


class MlDataPlotter:

    """
    The DataManager must contain the following data:
    self.data = data
    self.X_train_svd_df = X_train_svd_df
    self.y_train = y_train
    self.y_val = y_val
    self.y_pred = y_pred
    self.results = results
    self.symbol = symbol
    self.tc = tc
    """

    def __init__(self, data_manager, model_name, output_path="pdfs"):
        """
        Store performance data of a specific model test case.

        :param model_perf_data: Dictionary containing performance metrics of the model.
        """
        self.results_df = None
        self.model_name = model_name
        self.data_manager = data_manager
        self.output_path = output_path
        self.pdf = self._init_pdf()
        self._init_df()

    def _init_pdf(self):
        """Initialize a PDF at the given path with a filename based on the symbol."""
        directory = os.path.join(self.output_path, f"{self.model_name}")
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(self.output_path, f"{self.model_name}", "{self.data_manager.symbol}.pdf")
        pdf = PdfPages(file_path)
        return pdf

    def close_pdf(self):
        """Close the PDF after all plots are saved."""
        self.pdf.close()

    def _init_df(self):
        self.results_df = pd.DataFrame(columns=base_ml_data)

    def _save_plot(self):
        """Save the current plot to the PDF."""
        self.pdf.savefig(plt.gcf())
        plt.close()

    def store_testcase_data(self, model_perf_data):
        """
        Store performance data of a specific model test case.

        :param model_perf_data: Dictionary containing performance metrics of the model.
        """
        # Instead of appending the data to a list, we can directly use a dictionary for clarity
        testcase_data = {
            'symbol': model_perf_data["symbol"],
            'model_name': model_perf_data["model_name"],
            'acc_train': model_perf_data['accuracy_train'],
            'acc_val': model_perf_data['accuracy_val'],
            'precision': model_perf_data['precision'],
            'recall': model_perf_data['recall'],
            'f1': model_perf_data['f1'],
            'TN': model_perf_data['TN'],
            'FP': model_perf_data['FP'],
            'FN': model_perf_data['FN'],
            'TP': model_perf_data['TP'],
            'freq': model_perf_data["freq"],
            'len_train': model_perf_data["train_samples"],
            'len_val': model_perf_data["val_samples"],
            'aperf': model_perf_data["aperf"],
            'operf': model_perf_data["operf"],
            'num_trades': model_perf_data["num_trades"]
        }

        # Append the data to the dataframe.
        self.results_df = pd.concat([self.results_df, pd.DataFrame([testcase_data])], ignore_index=True)

    def store_data(self, output_dir, file_name):
        """
        Store the performance data to a CSV file.

        :param output_dir: Directory where the CSV will be saved.
        :param file_name: Name of the CSV file.
        """
        path = os.path.join(output_dir, file_name)
        self.results_df.to_csv(path, index=False)
        print(f"Stored the ML model performance data into {path}.")  # Replace print with logger if you use one.

    def _set_plot_properties(self, title, xlabel=None, ylabel=None, figsize=(8.3, 11.7), dpi=100):
        """
        Set properties for the matplotlib plot.

        :param title: Title of the plot.
        :param xlabel: Label for x-axis (optional).
        :param ylabel: Label for y-axis (optional).
        :param figsize: Size of the figure. Default is A4 size (8.3 x 11.7 inches).
        :param dpi: Dots per inch for plot quality. Default is 100.
        """
        plt.tight_layout()
        plt.figure(figsize=figsize, dpi=dpi)
        plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)

    def plot_explained_variance(self):
        """
        Plot the explained variance ratio for the SVD components and save the plot.
        """
        if not hasattr(self.data_manager, 'svd'):
            raise ValueError("You need to perform SVD first!")

        ncomps = self.data_manager.svd.n_components

        # Set plot properties using the utility method you have
        self._set_plot_properties('Explained Variance by Components')

        plt.plot(range(ncomps), self.data_manager.svd.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')

        # Set label sizes
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Cumulative Explained Variance', fontsize=12)

        # Set tick label sizes
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

        # Use ScalarFormatter to ensure standard formatting
        scalar_formatter = ScalarFormatter()
        scalar_formatter.set_scientific(False)
        plt.gca().yaxis.set_major_formatter(scalar_formatter)

        # Adjust precision if necessary
        y_formatter = FormatStrFormatter('%.2f')
        plt.gca().yaxis.set_major_formatter(y_formatter)

        plt.tight_layout()

        # Save the plot using the utility method you have
        self._save_plot()

    def plot_feature_importance(self, model):
        """
        Plot the importance of features.

        :param model: Trained machine learning model.
        """
        self._set_plot_properties('Importance of individual features for model predicting.')

        importances_length = len(model.feature_importances_)
        columns_length = len(self.data_manager.X_train.columns)

        # Ensure feature_importances_ length matches columns length
        if importances_length != columns_length:
            raise ValueError("Mismatch between feature importances length and number of columns.")

        importance = pd.DataFrame({'Importance': model.feature_importances_ * 100},
                                  index=self.data_manager.X_train.columns)

        importance.sort_values('Importance', ascending=True).plot(kind='barh', color='r', legend=False)

        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)

        # Set tick label sizes
        plt.xticks(fontsize=4)
        plt.yticks(fontsize=4)

        plt.tight_layout()

        self._save_plot()

    def plot_histogram(self):
        """Plot a histogram of the data."""
        original_settings = plt.rcParams.copy()
        plt.style.use('default')
        self._set_plot_properties('Data Histogram', figsize=(30, 30))
        axarr = self.data_manager.data.hist(sharex=False, sharey=False, xlabelsize=3, ylabelsize=3)

        # Adjust the title size for each subplot
        for ax in axarr.flatten():
            title = ax.get_title()
            ax.set_title(title, fontsize=3)  # you can change 10 to whatever size you prefer
            #ax.set_position([0.1, 0.1, 0.85, 0.85])  # [left, bottom, width, height]
        # Reduce the space between plots
        # Reduce the space between plots
        plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)  # Minimize padding
        self._save_plot()
        plt.rcParams.update(original_settings)

    def plot_signal_distribution(self):
        """Plot the distribution of signal values for training and validation datasets."""
        for dataset, label in [(self.data_manager.y_train, 'training'), (self.data_manager.y_test, 'validation')]:
            self._set_plot_properties(f'Signal values distribution of {label} dataset.')
            dataset.to_frame('signal').groupby(['signal']).size().plot(kind='barh', color='red')
            self._save_plot()

    def plot_covariance_matrix(self):
        """Plot the covariance matrix of the data."""
        self._set_plot_properties('Correlation Matrix', figsize=(30, 30))
        sns.heatmap(self.data_manager.data.corr(), vmax=1, square=True, annot=True, cmap='cubehelix')
        self._save_plot()

    def plot_tsne(self):
        """Perform and plot t-SNE dimensionality reduction on the data."""
        self._set_plot_properties(f"t-SNE dimensionality reduction")
        svdcols = [c for c in self.data_manager.X_train_svd_df.columns if c[0] == 'c']
        tsne_results = TSNE(n_components=2, random_state=0).fit_transform(self.data_manager.X_train_svd_df[svdcols])
        tsne_df = pd.DataFrame(tsne_results, columns=['x', 'y'], index=self.data_manager.X_train_svd_df.index)
        tsne_df['signal'] = self.data_manager.y_train
        sns.lmplot(data=tsne_df, x='x', y='y', hue='signal', fit_reg=False, scatter_kws={'alpha': 0.4, 's': 60},
                   height=8, aspect=1.5)
        self._save_plot()

    def plot_confusion_matrix(self):
        """Plot the confusion matrix comparing predicted and actual values."""
        self._set_plot_properties('Confusion Matrix')
        matrix = confusion_matrix(self.data_manager.y_test, self.data_manager.y_pred)
        df_cm = pd.DataFrame(matrix, columns=np.unique(self.data_manager.y_test),
                             index=np.unique(self.data_manager.y_test))
        df_cm.index.name, df_cm.columns.name = 'Actual', 'Predicted'
        sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})
        self._save_plot()

    def plot_model_evaluation(self):
        """
        Plot the evaluation of the model using different performance metrics.

        :param perf_metrics_dic: Dictionary containing performance metrics and their values.
        """
        # Extract metrics and values
        metrics_to_exclude = ['TN', 'FP', 'FN', 'TP']
        metrics = [metric for metric in self.data_manager.model_perf_data.keys() if metric not in metrics_to_exclude]
        values = [self.data_manager.model_perf_data[metric] for metric in metrics]
        self._set_plot_properties('Model Performance Metrics', figsize=(8.3, 11.7))

        plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
        plt.ylabel('Value')
        plt.ylim(0, 1)  # Assuming all metrics are between 0 and 1

        # Display the values on top of the bars
        for index, value in enumerate(values):
            plt.text(index, value + 0.02, str(value), ha='center')

        plt.tight_layout()
        self._save_plot()

    def plot_confusion_matrix_values(self):
        """
        Plot the values from the confusion matrix.

        :param perf_metrics_dic: Dictionary containing TN, FP, FN, and TP.
        """
        # Extract metrics and values
        metrics = ['TN', 'FP', 'FN', 'TP']
        values = [self.data_manager.model_perf_data[metric] for metric in metrics]

        self._set_plot_properties('Confusion Matrix Values', figsize=(8.3, 11.7))

        plt.bar(metrics, values, color=['blue', 'red', 'orange', 'green'])
        plt.ylabel('Counts')

        # Display the values on top of the bars
        for index, value in enumerate(values):
            plt.text(index, value + 0.02, str(value), ha='center')

        plt.tight_layout()
        self._save_plot()

    # def plot_model_evaluation(self, results, names):
    #     """
    #     Plot the evaluation of multiple algorithms/models.
    #
    #     :param results: List of results/performance metrics for each model.
    #     :param names: List of names corresponding to each model.
    #     """
    #     self._set_plot_properties('Algorithm Comparison', figsize=(15, 8))
    #     plt.boxplot(results)
    #     plt.xticks(list(range(1, len(names) + 1)), names)
    #     self._save_plot()

    def plot_backtest(self):
        """Plot backtesting results comparing creturns and cstrategy."""
        if not self.data_manager.results:
            print('No results to plot yet. Run a strategy.')
            return
        title = f'{self.data_manager.symbol} | TC = {self.data_manager.tc:.4f}'
        self.data_manager.results[['creturns', 'cstrategy']].plot(title=title, figsize=(10, 6))
        self._save_plot()
