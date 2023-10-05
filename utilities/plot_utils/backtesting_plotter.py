import os.path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utilities.logger import Logger
logger = Logger().get_logger()

base_str_data = ['ticker', 'indicator', 'dataset',  'num_trades', 'strat_multiple',
                 'bh_multiple', 'outperf_net', 'outperf', 'cagr', 'ann_mean', 'ann_std', 'sharpe', 'sortino',
                 'max_dd', 'calmar', 'max_dd_dur', 'kelly_crit', 'start_d',
                 'end_d', 'num_samples']

from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

class DataPlot:

    def __init__(self, dataset, indicator, symbol):
        self.df = None
        self.dataset = dataset
        self.indicator = indicator
        self.symbol = symbol
        self._init_df()
        #self.pdf = self._init_pdf()

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

    def _save_plot(self):
        """Save the current plot to the PDF."""
        self.pdf.savefig(plt.gcf())
        plt.close()

    def _init_df(self):
        self.df = pd.DataFrame()

    def store_testcase_data(self, perf_obj, params):
        if perf_obj.multiple_only:
            testcase_data = {
                'num_trades': perf_obj.num_of_trades,
                'strat_multiple': perf_obj.strategy_multiple,
                'bh_multiple': perf_obj.bh_multiple,
                'outperf_net': perf_obj.outperf_net,
                'outperf': perf_obj.outperf,
                'num_samples': perf_obj.num_samples
            }
        else:
            testcase_data = {
                'ticker': self.symbol,
                'indicator': self.indicator,
                'dataset': self.dataset,
                'num_trades': perf_obj.num_of_trades,
                'strat_multiple': perf_obj.strategy_multiple,
                'bh_multiple': perf_obj.bh_multiple,
                'outperf_net': perf_obj.outperf_net,
                'outperf': perf_obj.outperf,
                'cagr': perf_obj.cagr,
                'ann_mean': perf_obj.ann_mean,
                'ann_std': perf_obj.ann_std,
                'sharpe': perf_obj.sharpe,
                'sortino': perf_obj.sortino,
                'max_dd': perf_obj.max_drawdown,
                'calmar': perf_obj.calmar,
                'max_dd_dur': perf_obj.max_dd_duration,
                'kelly_crit': perf_obj.kelly_criterion,
                'start_d': perf_obj.start_d,
                'end_d': perf_obj.end_d,
                'num_samples': perf_obj.num_samples
            }

        # Merging the performance data with parameters
        testcase_data.update(params)
        # Append to the dataframe. If new columns are introduced in params, pandas will automatically create them.
        self.df = pd.concat([self.df, pd.DataFrame([testcase_data])], ignore_index=True)

    def store_data(self, output_dir):
        path = os.path.join(output_dir, f"{self.symbol}_{self.indicator}.csv")
        self.df.to_csv(path, index=False)
        logger.info(f"Stored the backtesting data into {path}.")

    def plot_results(self,results):
        """ Plots the performance of the trading strategy and compares to "buy and hold". """
        if results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} | Window = {} | Frequency = {} | TC = {}".format(self.symbol, self.window, self.freq, self.tc)
            results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))

        plt.tight_layout()

        # Save the plot using the utility method you have
        self._save_plot()