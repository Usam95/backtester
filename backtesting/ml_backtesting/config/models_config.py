from pydantic import BaseModel
from typing import List, Dict, Union, Optional


class TargetConfig(BaseModel):
    strategy: str
    N: Optional[int]
    threshold: Optional[float]


class ModelConfig(BaseModel):
    input_data_path: str
    output_data_path: str
    optimize: bool
    models: List[str]
    models_params: List[Dict[str, Union[str, Dict[str, Union[List[str], List[float], List[int]]]]]]
    target: TargetConfig


# Load and validate the configuration:
with open("classification_config.json", "r") as file:
    config_data = file.read()

config = ModelConfig.parse_raw(config_data)

