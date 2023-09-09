import os
import pandas as pd
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from utilities.logger import Logger

logger = Logger().get_logger()


class ConfigUpdater:

    def __init__(self, results_path="results/backward_h", config_file="forward_config.json"):
        # Define path to results folder, one level up from the current directory
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", results_path)
        self.config_file = config_file

    def get_max_outperf_config(self):
        # Define JSON structure
        data = {"strategies_config": {}}

        # Iterate through each folder in results
        for folder_name in os.listdir(self.results_path):
            folder_path = os.path.join(self.results_path, folder_name)

            # Check if it's a folder (like BTCUSDT, ETHUSDT, ...)
            if os.path.isdir(folder_path):

                # Set symbol name as key and initialize its value as an empty dict
                data["strategies_config"][folder_name] = {}

                # Iterate through each CSV file in the current folder
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".csv"):
                        file_path = os.path.join(folder_path, file_name)

                        # Read CSV file
                        df = pd.read_csv(file_path)

                        # Get row with max outperf_net value
                        max_outperf_row = df[df['outperf_net'] == df['outperf_net'].max()].iloc[0]

                        # Extract strategy name from file name
                        strategy_name = file_name.split("_")[1].split(".")[0].lower()

                        # Fill data according to strategy type using the values from the max_outperf_row
                        if strategy_name == "ema":
                            freq = int(max_outperf_row["freq"])
                            ema_s_val = int(max_outperf_row["ema_s"])
                            ema_l_val = int(max_outperf_row["ema_l"])

                            data["strategies_config"][folder_name][strategy_name] = {
                                "freq": freq,
                                "ema_s_val": ema_s_val,
                                "ema_l_val": ema_l_val
                            }
                            logger.info(f"Update symbol {folder_name} and {strategy_name=} with parameters {freq=}, {ema_s_val=}, {ema_l_val=}")

                        elif strategy_name == "sma":
                            freq = int(max_outperf_row["freq"])
                            sma_s_val = int(max_outperf_row["sma_s"])
                            sma_l_val = int(max_outperf_row["sma_l"])

                            data["strategies_config"][folder_name][strategy_name] = {
                                "freq": freq,
                                "sma_s_val": sma_s_val,
                                "sma_l_val": sma_l_val
                            }
                            logger.info(f"Update symbol {folder_name} and {strategy_name=} with parameters {freq=}, {sma_s_val=}, {sma_l_val=}")

                        elif strategy_name == "bb":
                            freq = int(max_outperf_row["freq"])
                            window = int(max_outperf_row["window"])
                            dev = int(max_outperf_row["dev"])

                            data["strategies_config"][folder_name][strategy_name] = {
                                "freq": freq,
                                "window": window,
                                "dev": dev
                            }
                            logger.info(f"Update symbol {folder_name} and {strategy_name=} with parameters {freq=}, {window=}, {dev=}")

                        elif strategy_name == "so":
                            freq = int(max_outperf_row["freq"])
                            k_period = int(max_outperf_row["k_period"])
                            d_period = int(max_outperf_row["d_period"])

                            data["strategies_config"][folder_name][strategy_name] = {
                                "freq": freq,
                                "k_period": k_period,
                                "d_period": d_period
                            }
                            logger.info(f"Update symbol {folder_name} and {strategy_name=} with parameters {freq=}, {k_period=}, {d_period=}")

                        elif strategy_name == "macd":
                            freq = int(max_outperf_row["freq"])
                            ema_s_val = int(max_outperf_row["ema_s_val"])
                            ema_l_val = int(max_outperf_row["ema_l_val"])
                            signal_mw = int(max_outperf_row["signal_mw"])

                            data["strategies_config"][folder_name][strategy_name] = {
                                "freq": freq,
                                "ema_s_val": ema_s_val,
                                "ema_l_val": ema_l_val,
                                "signal_mw": signal_mw
                            }
                            logger.info(f"Update symbol {folder_name} and {strategy_name=} with parameters {freq=}, {ema_s_val=}, {ema_l_val=}, {signal_mw=}")

                        elif strategy_name == "rsi":
                            freq = int(max_outperf_row["freq"])
                            periods = int(max_outperf_row["periods"])
                            rsi_upper = int(max_outperf_row["rsi_upper"])
                            rsi_lower = int(max_outperf_row["rsi_lower"])

                            data["strategies_config"][folder_name][strategy_name] = {
                                "freq": freq,
                                "periods": periods,
                                "rsi_upper": rsi_upper,
                                "rsi_lower": rsi_lower,

                            }
                            logger.info( f"Update symbol {folder_name} and {strategy_name=} with parameters {freq=}, {periods=}, {rsi_lower=}, {rsi_upper=}")
                        # ... repeat for other strategy types ...
        return data

    def update_config_file(self):
        new_data = self.get_max_outperf_config()

        # Load existing data
        with open(self.config_file, "r") as file:
            config_data = json.load(file)

        # Update strategies_config in the loaded data
        config_data["strategies_config"] = new_data["strategies_config"]

        # Save updated data to JSON file
        with open(self.config_file, "w") as outfile:
            json.dump(config_data, outfile, indent=4)


if __name__ == '__main__':
    updater = ConfigUpdater()
    updater.update_config_file()
