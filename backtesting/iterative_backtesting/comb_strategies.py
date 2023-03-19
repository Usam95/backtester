from abc import ABC, abstractmethod
import json

from config.config import config


print(config)


# Abstrakte Strategieklasse
class Strategy(ABC):
    @abstractmethod
    def execute(self):
        pass


class CombinedStrategy(Strategy):
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.strategies = []
        for strategy_config in config['strategies']:
            strategy_class = globals()[strategy_config['name']]
            strategy_params = strategy_config.get('params', {})
            strategy = strategy_class(**strategy_params)
            self.strategies.append(strategy)

    def execute(self, data):
        # Get the indicator values from each strategy
        sma = self.strategies[0].execute(data)
        ema = self.strategies[1].execute(data)
        rsi = self.strategies[2].execute(data)

        # Generate the combined buy/sell signal
        if sma > ema and rsi < 30:
            signal = "buy"
        elif sma < ema and rsi > 70:
            signal = "sell"
        else:
            signal = "hold"

        # Return the signal
        return signal