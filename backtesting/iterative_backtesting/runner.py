import json
# from ema import EMA
# from sma import SMA
# from rsi import RSI
# from so import SO
# from macd import MACD


class StrategyRunner:
    def __init__(self, config_file):
        self.config_file = config_file

    def run_strategies(self):
        with open(self.config_file) as f:
            config = json.load(f)

        strategies = config['strategies']
        for strategy in strategies:
            if strategy == 'EMA':
                period = config[strategy]['period']
                ema = EMA(period)
                ema.run()
            elif strategy == 'SMA':
                period = config[strategy]['period']
                sma = SMA(period)
                sma.run()
            elif strategy == 'RSI':
                period = config[strategy]['period']
                rsi = RSI(period)
                rsi.run()
            elif strategy == 'SO':
                period = config[strategy]['period']
                so = SO(period)
                so.run()
            elif strategy == 'MACD':
                fast_period = config[strategy]['fast_period']
                slow_period = config[strategy]['slow_period']
                signal_period = config[strategy]['signal_period']
                macd = MACD(fast_period, slow_period, signal_period)
                macd.run()
            else:
                print(f"Strategy '{strategy}' not recognized.")


# Beispiel-Konfigurationsdatei 'strategies.json'
# {
#     "strategies": ["EMA", "SMA", "RSI", "SO", "MACD"],
#     "EMA": {"period": 20},
#     "SMA": {"period": 50},
#     "RSI": {"period": 14},
#     "SO": {"period": 14},
#     "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
# }

# Beispielaufruf
runner = StrategyRunner('strategies.json')
runner.run_strategies()