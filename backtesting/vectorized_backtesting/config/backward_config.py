from pydantic import BaseModel
from typing import Optional
from json import load
import os

CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "backward_config.json")


class StrategiesConfig(BaseModel):
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


class BacktesterConfig(BaseModel):
    """
    All configuration relevant backtesting and optimizing.
    """

    strategies_config: Optional[StrategiesConfig]
    symbols: list[str]
    metric: str
    opt_method: str
    bayesian_trials: int
    hist_data_folder: str
    output_folder: str
    time_frame: dict[str, str]
    multiple_strategies: bool
    strategies: list[str]
    ptc: float


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

    strategies_config = StrategiesConfig(**parsed_conf["strategies_config"])
    parsed_conf.pop("strategies_config", None)

    backtester_conf = BacktesterConfig(strategies_config=strategies_config, **parsed_conf)

    return backtester_conf
