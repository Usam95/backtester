from pydantic import BaseModel
from typing import List, Dict, Union


class ModelConfig(BaseModel):
    model: str
    params: Dict[str, Union[float, List[Union[float, str, int]]]]


class ModelsConfig(BaseModel):
    models: List[ModelConfig]
