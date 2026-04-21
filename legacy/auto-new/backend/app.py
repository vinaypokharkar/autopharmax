
import logging
from fastapi import FastAPI
from backend.routes import router as prediction_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Inference API", version="1.0.0")

app.include_router(prediction_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Inference API...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Inference API...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
