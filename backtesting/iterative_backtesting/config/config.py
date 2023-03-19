
from pydantic import BaseModel
from typing import Optional
from json import load
import os

BACKTESTER_CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "backtester_config.json")
STRATEGY_CONFIG_FILES = os.path.join(os.path.dirname(os.path.realpath(__file__)), "strategy_config.json")

class StrategyConfig(BaseModel):
    """
    Configuration of technical indicator parameters
    to use for optimizing.
    """
    freq: int
    ema: dict[str, int]
    sma: dict[str, int]
    bb: dict[str, int]
    macd: dict[str, int]
    so: dict[str, int]
    rsi: dict[str, int]


class Backtester(BaseModel):
    """
    All configuration relevant backtesting and optimizing.
    """

    strategy_conf: Optional[StrategyConfig] = None
    symbol: str
    hist_data_folder: str
    time_frame: dict[str, str]
    strategies: list[str]
    ptc: float
    initial_balance: float
    units: float
    start_position: int
    take_profit: float
    stop_loss: float


def find_config_file(file_path):
    """Locate the configuration file."""
    print(file_path)
    if os.path.isfile(file_path):
        return file_path
    raise Exception(f"Config not found at {file_path}")


def fetch_config_from_yaml(cfg_path):
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file)
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config():
    """Run validation on config values."""
    # parse backtester configuration
    backtester_conf_file = find_config_file(BACKTESTER_CONFIG_FILE)
    backtester_parsed_conf = fetch_config_from_yaml(backtester_conf_file)

    # parse technical indicator configuration
    strategy_conf_file = find_config_file(STRATEGY_CONFIG_FILES)
    strategy_parsed_conf = fetch_config_from_yaml(strategy_conf_file)

    strategy_conf = StrategyConfig(**strategy_parsed_conf)
    backtester_conf = Backtester(**backtester_parsed_conf)

    backtester_conf.strategy_conf = strategy_conf

    return backtester_conf


config = create_and_validate_config()