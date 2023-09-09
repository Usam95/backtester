from pydantic import BaseModel
from typing import Optional
from json import load
import os

CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")


class StrategyConfig(BaseModel):
    """
    Configuration of technical indicator parameters
    to use for optimizing.
    """
    freq: list[int]
    ema: dict[str, list[int]]
    sma: dict[str, list[int]]
    bb: dict[str, list[int]]
    macd: dict[str, list[int]]
    so: dict[str, list[int]]
    rsi: dict[str, list[int]]


class SingleTestConfig(BaseModel):
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

    strategy_conf: Optional[StrategyConfig]
    single_test_conf: Optional[SingleTestConfig]
    tickers: list[str]
    single_test: bool
    metric: str
    opt_method: str
    bayesian_trials: int
    fetch_data: bool
    split_size: float
    hist_data_folder: str
    output_folder: str
    retrieve_data: bool
    time_frame: dict[str, str]
    multiple_strategies: bool
    strategies: list[str]
    ptc: float
    multiple_only: bool


def find_config_file(file_path):
    """Locate the configuration file."""
    if os.path.isfile(file_path):
        return file_path
    raise Exception(f"Config not found at {file_path}")


def fetch_config_from_json(cfg_path):
    """Parse JSON containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file)
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config():
    """Run validation on config values."""
    # parse the merged configuration
    conf_file = find_config_file(CONFIG_FILE)
    parsed_conf = fetch_config_from_json(conf_file)

    strategy_conf = StrategyConfig(**parsed_conf["strategy_conf"])
    parsed_conf.pop("strategy_conf", None)

    backtester_conf = Backtester(strategy_conf=strategy_conf, **parsed_conf)

    return backtester_conf


config = create_and_validate_config()