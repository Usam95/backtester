# Standard library imports
import os
import sys
from json import load
from typing import List, Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '', '..', '..')))
from utilities.logger import Logger

# Third-party library imports
from pydantic import BaseModel

CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")


class Strategy(BaseModel):
    ema: Optional[Dict[str, Any]]
    sma: Optional[Dict[str, Any]]
    # ..other strategies


class TickerConfig(BaseModel):
    ticker: str
    strategies: Strategy
    lookback_days: int
    bar_length: str
    units: int
    position: int
    stop_loss: float
    take_profit: float


class AppConfig(BaseModel):
    tickers_config: List[TickerConfig]


def fetch_config_from_file(cfg_path: str) -> Dict[str, Any]:
    """Parse JSON containing the configuration."""
    with open(cfg_path, "r") as conf_file:
        return load(conf_file)


def create_and_validate_config() -> AppConfig:
    """Read and validate configuration."""
    parsed_conf = fetch_config_from_file(CONFIG_FILE_PATH)
    config = AppConfig(**parsed_conf)
    logger = Logger().get_logger()
    logger.debug(f"Configured tickers: {[ticker.ticker for ticker in config.tickers_config]}")
    for ticker in config.tickers_config:
        logger.debug(f"Strategies for ticker {ticker.ticker}: {ticker.strategies}")
        logger.debug(f"Parameters for strategies of ticker {ticker.ticker}: EMA={ticker.strategies.ema}, SMA={ticker.strategies.sma}")
    return config
