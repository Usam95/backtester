import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utilities.logger import Logger

logger = Logger().get_logger()
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class BasisTester:

    def __init__(self, config):
        self.config = config
        self.check_dirs()

    def check_dirs(self):
        input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), self.config.hist_data_folder))
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), self.config.output_folder))

        # Check if historical data folder exists
        if not os.path.exists(self.config.hist_data_folder):
            logger.error(f"Provided historical data folder {self.config.hist_data_folder} is not found. Exiting..")
            exit(1)
        # Check if output folder exists, if not create it
        if not os.path.exists(self.config.output_folder):
            logger.warning(f"Provided output folder {self.config.output_folder} is not found. Creating it now.")
            try:
                os.makedirs(output_path)
                self.config.output_folder = output_path
            except OSError as e:
                logger.error(f"Failed to create the output folder. Reason: {e}.. Exiting..")
                exit(0)

    def create_ticker_output_dir(self, ticker):
        path = os.path.join(self.config.output_folder, ticker)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def get_hist_data_file(self, ticker):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), self.config.hist_data_folder, ticker,
                                            f"{ticker}.parquet.gzip"))
        if not os.path.exists(path):
            logger.error(f"Path {path} for historical data not found.\n"
                         f"Configure data retrieving or provide the correct path.")
            exit(0)
        return path

    def print_exec_time(self, start_t, end_t, tech_ind, ticker):
        total_time = round(((end_t - start_t) / 60), 2)
        logger.info(f"Optimized {tech_ind.upper()} for {ticker} in {total_time} minutes.")

    def print_message(self, msg, ticker):
        logger.info("*" * 100)
        logger.info(f"Started {msg}-based backtesting for {ticker}.")
        logger.info("*" * 100)

    def generate_title(self, symbol, strategy_name, perf_data, **strategy_params):
        base_title = f"{symbol} | "
        base_title += f"{strategy_name} : ("
        param_titles = [f"{key} = {value}" for key, value in strategy_params.items()]
        param_string = base_title + " , ".join(param_titles) + " )"

        perf_titles = [f"{key} = {value}" for key, value in perf_data.items()]
        perf_string = " , ".join(perf_titles)

        return param_string + "\n" + perf_string

    def plot_performance_to_axis(self, perf_obj, ax, strategy_name, symbol, **strategy_params):
        perf_data = {
            "Strategy Multiple": perf_obj.strategy_multiple,
            "Buy/Hold Multiple": perf_obj.bh_multiple,
            "Net Out-/Under Perf ": perf_obj.outperf_net,
            "Num Trades": perf_obj.num_of_trades
        }

        title = self.generate_title(symbol, strategy_name, perf_data, **strategy_params)

        perf_obj.data[["creturns", "cstrategy"]].plot(ax=ax)
        ax.set_xlabel('Date', fontsize=8)
        ax.set_ylabel("Performance", fontsize=8)

        ax.legend(["creturns", "cstrategy"], fontsize=6)
        # Adjust the title's font and placement to better fit LaTeX style
        ax.figure.text(0.5, 1.07, title, ha='center', va='center', fontsize=6, transform=ax.transAxes, family='DejaVu Serif')
        # Adjust the tick label size for both x and y axes
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

    def plot_results(self, perf_obj, pdf, strategy_name, symbol, **strategy_params):
        """ Plots the performance of the trading strategy and compares to "buy and hold". """
        if perf_obj.data is None:
            return

        self.combined_plot_to_pdf(perf_obj, pdf, strategy_name, symbol, **strategy_params)

        #self.combined_plot_to_pdf(perf_obj.data, pdf)

    def plot_trading_signals_to_axis(self, data_df, ax):
        ax.plot(data_df.index, data_df["Close"])

        # Initialize previous signal variable
        prev_signal = None

        # Add arrows for buy (1) and sell (0) signals
        for i, (date, row) in enumerate(data_df.iterrows()):
            if row["trades"] == 1 and prev_signal != 1:
                ax.plot(date, row["Close"], "^g", markersize=4)
                prev_signal = 1
            elif row["trades"] == 0 and prev_signal != 0:
                ax.plot(date, row["Close"], "vr", markersize=4)
                prev_signal = 0

        # Add labels and legend
        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel("Close Prices", fontsize=8)
        ax.legend(["Close", "Buy", "Sell"], fontsize=6)

        # Adjust the tick label size for both x and y axes
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.set_title("Trading Signals", fontsize=8)

    def combined_plot_to_pdf(self, perf_obj, pdf, strategy_name, symbol, **strategy_params):
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6,9))

        # First plot (performance)
        self.plot_performance_to_axis(perf_obj, axes[0], strategy_name, symbol, **strategy_params)

        # Second plot (trading signals)
        self.plot_trading_signals_to_axis(perf_obj.data, axes[1])

        # Third plot (market position)
        self.plot_market_position_to_axis(perf_obj.data, axes[2])

        # Save the combined plot to the PDF
        plt.tight_layout()
        #pdf.savefig(fig, bbox_inches='tight')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def plot_market_position_to_axis(self, data, ax):
        data['position'].plot(ax=ax, ylim=[-0.1, 1.1])
        ax.set_ylabel("Position", fontsize=8)
        ax.set_xlabel("Date", fontsize=8)
        # Adjust the tick label size for both x and y axes
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

        ax.set_title("Market Positioning", fontsize=8)
