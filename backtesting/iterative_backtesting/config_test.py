
import json
import os
import pandas as pd
import itertools

from config.config import config
from strategies import SMA
from strategies import EMA
from strategies import MACD
from strategies import RSI
from strategies import SO
from strategies import BB

# Load the DataFrame
filepath = os.path.abspath(os.path.join("..", "..", "hist_data", "XRPUSDT", "XRPUSDT.parquet.gzip"))
data = pd.read_parquet(filepath)

# Create a dictionary with parameters of technical indicators
tech_indic_list = dict()
with open("config/strategy_config.json", 'r') as f:
    tech_indic_list = json.load(f)

# Create a list with Technical Indicator instances
tech_indic_objs = []
for strategy in config.strategies:
    tech_ind_class = globals()[strategy.upper()]
    tech_ind = tech_ind_class(**tech_indic_list[strategy])
    tech_indic_objs.append(tech_ind)

# Execute the strategies
#for tech_indicator in tech_indic_objs:
#    print(tech_indicator.get_signal(data))


class StrategyExecutor:

    def __init__(self, data, strategy_names, strategy_params):
        self.data = data
        self.strategy_param_dic = strategy_params
        self.strategy_names = strategy_names
        self.strategies_instances = {}
        self.strategy_combinations = []

        self.init_combinations()
        self.init_strategy_instances()

    def init_combinations(self):
        for i in range(1, len(self.strategy_names) + 1):
            for subset in itertools.combinations(self.strategy_names, i):
                self.strategy_combinations.append(subset)

    def init_strategy_instances(self):
        for combination in self.strategy_combinations:
            for strategy in combination:
                if strategy == "rsi":
                    self.strategies_instances[strategy] = RSI(**self.strategy_param_dic[strategy.lower()])
                elif strategy == "sma":
                    self.strategies_instances[strategy] = SMA(**self.strategy_param_dic[strategy.lower()])
                elif strategy == "ema":
                    self.strategies_instances[strategy] = EMA(**self.strategy_param_dict[strategy.lower()])
                elif strategy == "macd":
                    self.strategies_instances[strategy] = MACD(**self.strategy_param_dic[strategy.lower()])
                elif strategy == "so":
                    self.strategies_instances[strategy] = SO(**self.strategy_param_dic[strategy.lower()])
                elif strategy == "bb":
                    self.strategies_instances[strategy] = BB(**self.strategy_param_dic[strategy.lower()])

    def execute_strategies(self):
        # Initialize buy and sell signals
        buy_signals = ["Buy"] * len(self.strategy_combinations)
        sell_signals = ["Sell"] * len(self.strategy_combinations)

        # Calculate the maximum number of rows required for all strategies
        max_periods = max([strategy.periods for strategy in self.strategies_instances.values()])

        # Iterate over the rows in the data DataFrame
        for i, row in self.data.iterrows():
            # Iterate over the strategy combinations
            for j, combination in enumerate(self.strategy_combinations):
                # Iterate over the individual strategies in the combination
                for k, strategy in enumerate(combination):
                    # Get the appropriate number of rows from the data DataFrame for the strategy's computation
                    strategy_instance = self.strategies_instances[strategy]
                    num_rows = max_periods + 1
                    start_row = max(i - num_rows + 1, 0)
                    end_row = i + 1

                    # Compute the signal for the strategy
                    signal = strategy_instance.get_signal(self.data.iloc[start_row:end_row])
                    if signal == "Buy":
                        sell_signals[j] = ""
                    elif signal == "Sell":
                        buy_signals[j] = ""

            # Check if we should buy or sell based on the signals
            if all(signal == "Buy" for signal in buy_signals):
                print("Buy at {}".format(row["Close"]))
                # Place buy order
                # Reset signals
                buy_signals = ["Buy"] * len(self.strategy_combinations)
                sell_signals = ["Sell"] * len(self.strategy_combinations)
            elif all(signal == "Sell" for signal in sell_signals):
                print("Sell at {}".format(row["Close"]))
                # Place sell order
                # Reset signals
                buy_signals = ["Buy"] * len(self.strategy_combinations)
                sell_signals = ["Sell"] * len(self.strategy_combinations)