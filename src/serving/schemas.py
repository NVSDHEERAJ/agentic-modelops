from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Annotated

class PredictionRequest(BaseModel):
    features: Annotated[Dict[str, Any], Field(..., description = "Dictionary of feature names and their corresponding values")]
    actual_label: Optional[int] = None # Optional field for actual label, useful for monitoring and evaluation

class PredictionResponse(BaseModel):
    prediction: int
    fraud_probability: float
    threshold: float
    model_version: str
    deployment_id: Optional[int] = None
    deployed_at: Optional[str] = None