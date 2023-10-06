
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
#sns.set_style("whitegrid")
#sns.set_context("talk")  # this will make plot elements slightly larger

base_ml_data = [
    'symbol', 'model_name', 'acc_train', 'acc_val', 'precision',
    'recall', 'f1', 'TN', 'FP', 'FN', 'TP', 'freq', 'len_train',
    'len_val',  'aperf', 'operf', 'num_trades'
]

from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
plt.rcParams["font.size"] = 8

#sns.set_style("whitegrid")
sns.set_context("talk")


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

    def __init__(self, data_manager, model_name, target, output_path="pdfs"):
        """
        Store performance data of a specific model test case.

        :param model_perf_data: Dictionary containing performance metrics of the model.
        """
        self.target = target
        self.results_df = None
        self.model_name = model_name
        self.data_manager = data_manager
        self.output_path = output_path
        self.pdf = self._init_pdf()
        self._init_df()

    def _init_pdf(self):
        """Initialize a PDF at the given path with a filename based on the symbol."""
        directory = os.path.join(self.output_path, f"{self.model_name}", f"{self.data_manager.symbol}")
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(self.output_path, f"{self.model_name}", f"{self.data_manager.symbol}", f"{self.target}.pdf")
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
        #self._set_plot_properties(figsize=(8, 6))
        plt.figure(figsize=(8, 4))
        plt.plot(range(ncomps), self.data_manager.svd.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')

        # Set label sizes
        plt.xlabel('Number of Components', fontsize=10)
        plt.ylabel('Cumulative Explained Variance', fontsize=10)

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
        #self._set_plot_properties('Relative Importance of Features in Predictive Model')

        importances_length = len(model.feature_importances_)
        columns_length = len(self.data_manager.feature_columns)

        color = sns.color_palette("pastel", n_colors=1)

        # Ensure feature_importances_ length matches columns length
        if importances_length != columns_length:
            raise ValueError("Mismatch between feature importances length and number of columns.")

        importance = pd.DataFrame({'Importance': model.feature_importances_ * 100},
                                  index=self.data_manager.feature_columns)

        importance.sort_values('Importance', ascending=True).plot(kind='barh', color=color, legend=False)

        #plt.title("Contribution of Features to Model's Prediction", fontsize=8, y=1.01)
        plt.xlabel('Importance [in %]', fontsize=8)
        plt.ylabel('Features', fontsize=8)

        # Set tick label sizes
        plt.xticks(fontsize=4)
        plt.yticks(fontsize=4)

        plt.tight_layout()

        self._save_plot()

    def plot_histogram(self):
        """Plot a histogram of the data."""
        original_settings = plt.rcParams.copy()
        plt.style.use('default')
        self._set_plot_properties('Data Histogram', figsize=(8, 6))
        axarr = self.data_manager.data.hist(sharex=False, sharey=False, xlabelsize=3, ylabelsize=3)

        # Adjust the title size for each subplot
        for ax in axarr.flatten():
            ax.ticklabel_format(useOffset=False, style='plain', axis='both')  # prevent scientific notation
            title = ax.get_title()
            ax.set_title(title, fontsize=3)  # you can change 10 to whatever size you prefer
            #ax.set_position([0.1, 0.1, 0.85, 0.85])  # [left, bottom, width, height]
        # Reduce the space between plots
        # Reduce the space between plots
        plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)  # Minimize padding
        self._save_plot()
        plt.rcParams.update(original_settings)

    def plot_signal_distribution(self):
        unique_signals = self.data_manager.y_train.unique()
        bright_green, bright_blue = sns.color_palette("pastel", n_colors=2)

        # Set figure size
        plt.figure(figsize=(8, 4))

        # Set plot properties
        plt.ylabel('Target', fontsize=8)
        plt.xlabel('Number of target values', fontsize=8)

        # Get the data
        train_data = self.data_manager.y_train.value_counts().reindex(unique_signals, fill_value=0)
        validation_data = self.data_manager.y_test.value_counts().reindex(unique_signals, fill_value=0)

        # Plot the data
        bar_width = 0.35
        indices = range(len(unique_signals))

        bars1 = plt.barh([i - bar_width / 2 for i in indices], train_data, height=bar_width, color=bright_green,
                         label='Training')
        bars2 = plt.barh([i + bar_width / 2 for i in indices], validation_data, height=bar_width, color=bright_blue,
                         label='Validation')

        plt.yticks(indices, unique_signals, fontsize=8)
        plt.xticks(fontsize=8)
        plt.legend(fontsize=8)

        # Annotate the bars with their values
        for bar in bars1:
            y = bar.get_y() + bar.get_height() / 2
            x = bar.get_width()
            plt.text(x, y, str(int(x)), va='center', ha='left', fontsize=8, color='black')

        for bar in bars2:
            y = bar.get_y() + bar.get_height() / 2
            x = bar.get_width()
            plt.text(x, y, str(int(x)), va='center', ha='left', fontsize=8, color='black')

        # Adjust the subplot parameters to remove white space
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

        self._save_plot()

    def plot_covariance_matrix(self):
        """Plot the covariance matrix of the data."""
        self._set_plot_properties('Correlation Matrix', figsize=(30, 30))

        # Set the title with increased font size and adjust the position
        plt.title('Correlation Matrix', fontsize=20, y=1.02)

        # Choose a different heatmap color palette. For instance, 'YlGnBu'
        sns.heatmap(self.data_manager.data.corr(), vmax=1, square=True, annot=True, cmap='Greens')

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

    def plot_classification_report(self, train=False):
        """Plot the confusion matrix and classification report metrics side by side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.85))

        # Adjust space between plots
        plt.subplots_adjust(hspace=0.5)  # adjust this value as needed

        if train:
            matrix = confusion_matrix(self.data_manager.y_train, self.data_manager.y_pred_train)
        else:
            matrix = confusion_matrix(self.data_manager.y_test, self.data_manager.y_pred_test)

        # Create custom labels
        labels = [["TN", "FP"], ["FN", "TP"]]
        new_labels = [["{0}\n{1:d}".format(label, value) for label, value in zip(row_label, row_value)]
                      for row_label, row_value in zip(labels, matrix)]

        df_cm = pd.DataFrame(matrix, columns=np.unique(self.data_manager.y_test),
                             index=np.unique(self.data_manager.y_test))
        df_cm.index.name, df_cm.columns.name = 'Actual', 'Predicted'

        # Plot the confusion matrix heatmap
        sns.heatmap(df_cm, cmap="Blues", annot=new_labels, fmt='', annot_kws={"size": 16}, ax=ax1)
        ax1.set_title('Confusion Matrix')

        # Plot the performance metrics
        self.plot_performance_metrics(matrix, ax2)

        plt.tight_layout()
        self._save_plot()

    def plot_performance_metrics(self, matrix, ax):
        """Plot classification performance metrics as a bar plot."""
        #bright_green, bright_blue = sns.color_palette("icefire", n_colors=2)

        # Compute metrics
        TN, FP, FN, TP = matrix.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)

        metrics = [accuracy, precision, recall, f1_score]
        labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

        # Colors from the icefire palette
        colors = sns.color_palette("pastel", n_colors=4)

        # Plot the vertical bar plot using the generated colors
        bars = ax.bar(labels, metrics, color=[colors[0], colors[1], colors[2], colors[3]])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        #ax.set_xlabel('Model Performance')
        ax.set_title('Performance Metrics')
        #ax.xaxis.set_label_coords(0.5, -0.8)  # adjust as needed

        # Adding data labels on top of the bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, round(yval, 2), ha='center', va='bottom',
                    fontsize=12)


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

    def _generate_title(self, symbol, strategy_name, perf_data, **strategy_params):
        # Center align model name
        base_title = f"{symbol} | {strategy_name}".center(80)  # assuming you want to center it over an 80-char width

        # Convert strategy_params into titles, filtering out ones with value of None
        param_titles = [f"{key} = {value}" for key, value in strategy_params.items() if value is not None]

        # Group parameters into chunks of 4
        param_chunks = [param_titles[i:i + 6] for i in range(0, len(param_titles), 6)]
        param_lines = [" , ".join(chunk) for chunk in param_chunks]

        # Convert perf_data into titles, again filtering out None values
        perf_titles = [f"{key} = {value}" for key, value in perf_data.items() if value is not None]

        # Construct the final title string
        title = base_title  # Centered model name
        title += "\n\nModel Parameters:\n"  # Model Parameters title
        for line in param_lines:
            title += '\t' + line + "\n" # Adding parameters 4 per line
        # Add performance data
        title += "\nPerformance:\n\t[" + " , ".join(perf_titles) + "]"

        return title

    def plot_performance_to_axis(self, data_subset, perf_obj, config, **strategy_params):
        # Create a new figure and axis
        plt.figure(figsize=(6, 4))

        # Get performance data from the perf_obj
        perf_data = {
            "Strategy Multiple": perf_obj.strategy_multiple_net,
            "Buy/Hold Multiple": perf_obj.bh_multiple,
            "Net Out-/Under Perf": perf_obj.outperf_net,
            "Num Trades": perf_obj.num_of_trades
        }

        # Plotting
        data_subset[["creturns", "cstrategy_net"]].plot()
        title = self._generate_title(config.dataset_conf.symbol, config.model_name, perf_data, **strategy_params)
        plt.title(title, fontsize=6, family='DejaVu Serif', loc="left")  # setting the title for the plot

        plt.xlabel('Date', fontsize=8)
        plt.ylabel("Performance", fontsize=8)
        plt.legend(["creturns", "cstrategy_net"], fontsize=7, loc="upper right")
        plt.tick_params(axis='x', labelsize=6)
        plt.tick_params(axis='y', labelsize=6)

        # If mode is "both", plot a vertical line to show where training data ends and test data begins
        if config.dataset_conf.mode == "full" and config.dataset_conf.split_date:
            plt.axvline(x=config.dataset_conf.split_date, color='red', linestyle='--', linewidth=0.5)
        # Adjust margins
        plt.subplots_adjust(top=0.81, bottom=0.1)
        # Save the plot into the PDF
        self._save_plot()

    def plot_backtest(self):
        """Plot backtesting results comparing creturns and cstrategy."""
        if not self.data_manager.results:
            print('No results to plot yet. Run a strategy.')
            return
        title = f'{self.data_manager.symbol} | TC = {self.data_manager.tc:.4f}'
        self.data_manager.results[['creturns', 'cstrategy_net']].plot(title=title, figsize=(10, 6))
        self._save_plot()
