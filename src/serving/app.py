from fastapi import FastAPI, HTTPException

from src.serving.schemas import PredictionRequest, PredictionResponse
from src.serving.model_service import FraudModelService

app = FastAPI(
    title = "Fraud Dectection API",
    description = "FastAPI service for IEEE fraud detection model inference",
    version = "1.0.0",
)

model_service = FraudModelService()

@app.get("/")
def root():
    return {"message" : "Fraud Dectection API is up and running!"}

@app.get("/health")
def health_check():
    return {
        "status" : "healthy",
        "model_loaded" : model_service.model is not None,
        "model_version" : model_service.model_version,
    }

@app.post("/predict", response_model = PredictionResponse)
def predict(request : PredictionRequest):
    try:
        result = model_service.predict(request.features)
        return result
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Prediction failed: {str(e)}")