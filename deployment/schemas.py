from pydantic import BaseModel
from typing import List


class PredictionRequest(BaseModel):
    instances: List[List[float]]


class PredictionResponse(BaseModel):
    predictions: List[int]
