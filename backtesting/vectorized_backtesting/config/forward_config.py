from pydantic import BaseModel
from typing import Optional, Dict
from json import load
import os

CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "forward_config.json")


class Strategy(BaseModel):
    ema: Dict[str, int]
    sma: Dict[str, int]
    bb: Dict[str, int]
    macd: Dict[str, int]
    so: Dict[str, int]
    rsi: Dict[str, int]


class StrategiesConfig(BaseModel):
    strategies_config: Dict[str, Strategy]


class ForwardTester(BaseModel):
    """
    All configuration relevant backtesting and optimizing.
    """
    strategies_config: Optional[StrategiesConfig]
    symbols: list[str]
    metric: str
    hist_data_folder: str
    output_folder: str
    time_frame: dict[str, str]
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

    # Instead of expanding the strategies_config, directly pass it as it is
    strategies_config = StrategiesConfig(strategies_config=parsed_conf["strategies_config"])
    parsed_conf.pop("strategies_config", None)

    forward_tester_conf = ForwardTester(strategies_config=strategies_config, **parsed_conf)

    return forward_tester_conf
