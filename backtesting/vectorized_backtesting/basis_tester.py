import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from utilities.logger import Logger

logger = Logger().get_logger()


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

    def plot_results(self, perf_obj, pdf, strategy_name, symbol, **strategy_params):
        """ Plots the performance of the trading strategy and compares to "buy and hold". """
        if perf_obj.data is None:
            return

        perf_data = {
            "Strategy Multiple": perf_obj.strategy_multiple,
            "Buy/Hold Multiple": perf_obj.bh_multiple,
            "Net Out-/Under Perf ": perf_obj.outperf_net,
            "Num Trades": perf_obj.num_of_trades
        }

        title = self.generate_title(symbol, strategy_name, perf_data, **strategy_params)

        ax = perf_obj.data[["creturns", "cstrategy"]].plot(figsize=(12, 8))
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel("Performance")
        fig = ax.get_figure()
        fig.text(0.5, -0.15, title, ha='center', va='center', fontsize=8, transform=ax.transAxes)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Add the trading activity plot to the PDF
        self.plot_trading_activity(perf_obj.data, pdf)

    def plot_trading_activity(self, data_df, pdf):
        plt.figure(figsize=(20, 12))
        plt.plot(data_df.index, data_df["Close"])

        # Initialize previous signal variable
        prev_signal = None

        # Add arrows for buy (1) and sell (0) signals
        for i, (date, row) in enumerate(data_df.iterrows()):
            if row["trades"] == 1 and prev_signal != 1:
                plt.plot(date, row["Close"], "^g", markersize=10)
                prev_signal = 1
            elif row["trades"] == 0 and prev_signal != 0:
                plt.plot(date, row["Close"], "vr", markersize=10)
                prev_signal = 0

        # Add labels and legend
        plt.xlabel("Date")
        plt.ylabel("Close Prices")
        plt.legend(["Close", "Buy", "Sell"])

        # Save the plot to the PDF
        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

