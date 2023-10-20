from typing import List, Dict, Union, Optional
from pydantic import BaseModel


class TargetConfig(BaseModel):
    target: str
    N: Optional[int]
    threshold: Optional[float]


class DataSet(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    split_date: str
    freq: str
    mode: str
    target_conf: TargetConfig


class ModelConfig(BaseModel):
    model: str
    params: Dict[str, Union[float, List[Union[float, str, int]]]]


class BacktesterConfig(BaseModel):
    tc: float
    dataset_conf: DataSet
    models: List[str]
    models_config: List[ModelConfig]
    model_name: str
    model_type: str
