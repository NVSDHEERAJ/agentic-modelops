from fastapi import FastAPI, HTTPException

from src.serving.schemas import PredictionRequest, PredictionResponse
from src.serving.model_service import FraudModelService

from src.logging.db import DatabaseManager
from src.logging.prediction_logger import PredictionLogger

db_manager = DatabaseManager()
db_manager.initialize_database()

prediction_logger = PredictionLogger(db_manager)

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
        "deployment_id" : model_service.deployment_id,
        "deployed_at" : model_service.deployed_at
    }

@app.post("/predict", response_model = PredictionResponse)
def predict(request : PredictionRequest):
    try:
        result = model_service.predict(request.features)

        prediction_logger.log_prediction(
            features = request.features,
            fraud_probability = result["fraud_probability"],
            prediction = result["prediction"],
            threshold = result["threshold"],
            model_version = result["model_version"],
            deployment_id = result["deployment_id"],
            actual_label = request.actual_label
        )

        return result
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Prediction failed: {str(e)}")