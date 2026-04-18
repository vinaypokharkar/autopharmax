from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Optional, Dict
from services.model_service import ModelService
from services.data_service import DataService

router = APIRouter()

# Initialize services
model_service = ModelService()
data_service = DataService()

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================
class PredictionRequest(BaseModel):
    cell_line: str
    drug_name: str

class PredictionResponse(BaseModel):
    predicted_ln_ic50: float
    predicted_ic50: float
    actual_ln_ic50: Optional[float] = None
    actual_ic50: Optional[float] = None
    absolute_error: Optional[float] = None
    shap_values: Optional[Dict[str, float]] = None
    model_name: str
    training_date: str
    num_features: int

class AvailableOptionsResponse(BaseModel):
    cell_lines: list[str]
    drugs: list[str]

# ============================================
# ROUTES
# ============================================

@router.get("/options", response_model=AvailableOptionsResponse)
async def get_available_options():
    """
    Get available cell lines and drugs for prediction
    """
    try:
        original_data = data_service.get_original_data()
        
        cell_lines = sorted(original_data['TCGA_DESC'].unique().tolist())
        drugs = sorted(original_data['DRUG_NAME'].unique().tolist()) if 'DRUG_NAME' in original_data.columns else []
        
        return AvailableOptionsResponse(
            cell_lines=cell_lines,
            drugs=drugs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching options: {str(e)}")


@router.post("/predict", response_model=PredictionResponse)
async def predict_ic50(request: PredictionRequest):
    """
    Predict IC50 value for a given cell line and drug combination
    """
    try:
        # Get model, scaler, and metadata
        model = model_service.get_model()
        scaler = model_service.get_scaler()
        metadata = model_service.get_metadata()
        
        # Get original data
        original_data = data_service.get_original_data()
        
        # Filter data for selected combination
        filtered_data = original_data[
            (original_data['TCGA_DESC'] == request.cell_line) &
            (original_data['DRUG_NAME'] == request.drug_name)
        ]
        
        if len(filtered_data) == 0:
            # Use mean values for features if combination not found
            feature_data = original_data[metadata['feature_columns']].mean().to_frame().T
            has_actual_value = False
        else:
            # Get the first matching row's features
            feature_data = filtered_data[metadata['feature_columns']].iloc[0:1]
            has_actual_value = True
        
        # Scale features
        X_scaled = scaler.transform(feature_data)
        
        # Predict
        ln_ic50_pred = model.predict(X_scaled)[0]
        ic50_pred = np.exp(ln_ic50_pred)  # Convert from log scale
        
        # Calculate SHAP values
        try:
            explainer = model_service.get_explainer()
            shap_output = explainer.shap_values(X_scaled)
            
            # If multi-dimensional, take the first sample's SHAP values
            # (In TreeExplainer for regression, it's usually a single array)
            if isinstance(shap_output, list):
                shap_contribs = shap_output[0].flatten()
            else:
                shap_contribs = shap_output.flatten()
            
            # Map features to their SHAP values
            feature_names = metadata['feature_columns']
            shap_dict = {name: float(val) for name, val in zip(feature_names, shap_contribs)}
        except Exception as e:
            print(f"SHAP calculation failed: {e}")
            shap_dict = None

        # Prepare response
        response_data = {
            "predicted_ln_ic50": float(ln_ic50_pred),
            "predicted_ic50": float(ic50_pred),
            "shap_values": shap_dict,
            "model_name": metadata['best_model_name'],
            "training_date": metadata['training_date'],
            "num_features": len(metadata['feature_columns'])
        }
        
        # Add actual values if available
        if has_actual_value and 'LN_IC50' in filtered_data.columns:
            actual_ln_ic50 = filtered_data['LN_IC50'].iloc[0]
            actual_ic50 = np.exp(actual_ln_ic50)
            error = abs(ln_ic50_pred - actual_ln_ic50)
            
            response_data.update({
                "actual_ln_ic50": float(actual_ln_ic50),
                "actual_ic50": float(actual_ic50),
                "absolute_error": float(error)
            })
        
        return PredictionResponse(**response_data)
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid cell line or drug name: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify model and data are loaded
    """
    try:
        model_loaded = model_service.get_model() is not None
        data_loaded = data_service.get_original_data() is not None
        
        return {
            "status": "healthy" if (model_loaded and data_loaded) else "unhealthy",
            "model_loaded": model_loaded,
            "data_loaded": data_loaded
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")