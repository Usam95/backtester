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
    freq: list[int]
    metric: str
    ema: dict[str, list[int]]
    sma: dict[str, list[int]]
    bb: dict[str, list[int]]
    macd: dict[str, list[int]]
    so: dict[str, list[int]]
    rsi: dict[str, list[int]]


class Backtester(BaseModel):
    """
    All configuration relevant backtesting and optimizing.
    """

    strategy_conf: Optional[StrategyConfig] = None
    tickers: list[str]
    fetch_data: bool
    split_size: float
    hist_data_folder: str
    output_folder: str
    retrieve_data: bool
    time_frame: dict[str, str]
    multiple_strategies: bool
    ml_strategy: bool
    strategies: list[str]
    ptc: float
    multiple_only: bool



def find_config_file(file_path):
    """Locate the configuration file."""
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