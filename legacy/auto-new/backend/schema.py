from pydantic import BaseModel
from typing import Optional, Dict

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add other features as needed

class PredictionResponse(BaseModel):
    prediction: float
    probability: Optional[float] = None
    status: str
