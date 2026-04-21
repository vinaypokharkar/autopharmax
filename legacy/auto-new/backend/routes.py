
import logging
from fastapi import APIRouter, HTTPException
from backend.schema import PredictionRequest, PredictionResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Placeholder for loading only the inference model
# from src.models.predict import load_model, predict
# model = load_model()

@router.get("/health")
def health_check():
    return {"status": "ok", "service": "inference-api"}

@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    logger.info(f"Prediction request: {request}")
    try:
        # result = model.predict(request)
        # Dummy result
        result = 0.85
        return PredictionResponse(prediction=result, status="success")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
