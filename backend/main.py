import sys
import os

# Add backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.predict import router as predict_router

# Create FastAPI app
app = FastAPI(
    title="Drug Response Predictor API",
    description="API for predicting IC50 values for cancer drug combinations",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router, prefix="/api/v1", tags=["Prediction"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Drug Response Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/health",
            "options": "/api/v1/options",
            "predict": "/api/v1/predict"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)