import pickle
import os
import shap
from typing import Optional

class ModelService:
    """
    Service for loading and managing ML models
    """
    _instance = None
    _model = None
    _scaler = None
    _metadata = None
    _explainer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._load_models()
        return cls._instance
    
    def _load_models(self):
        """Load trained model, scaler, metadata, and init SHAP explainer"""
        # ... existing loading code ...
        # Use absolute path relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_dir = os.path.join(base_dir, "models")

        
        try:
            # Load best model
            model_path = os.path.join(model_dir, 'best_model_XGBoost_(Tuned).pkl')
            with open(model_path, 'rb') as f:
                self._model = pickle.load(f)
            
            # Load scaler
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self._scaler = pickle.load(f)
            
            # Load metadata
            metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
            with open(metadata_path, 'rb') as f:
                self._metadata = pickle.load(f)
            
            # Initialize SHAP explainer
            self._explainer = shap.TreeExplainer(self._model)
            
            print("✅ Models and SHAP explainer loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print(f"Model directory: {model_dir}")
            if os.path.exists(model_dir):
                print(f"Files in directory: {os.listdir(model_dir)}")
            else:
                print("Directory not found")
            raise
    
    def get_model(self):
        """Get the loaded model"""
        if self._model is None:
            raise ValueError("Model not loaded")
        return self._model
    
    def get_scaler(self):
        """Get the loaded scaler"""
        if self._scaler is None:
            raise ValueError("Scaler not loaded")
        return self._scaler
    
    def get_metadata(self):
        """Get the model metadata"""
        if self._metadata is None:
            raise ValueError("Metadata not loaded")
        return self._metadata

    def get_explainer(self):
        """Get the SHAP explainer"""
        if self._explainer is None:
            raise ValueError("SHAP explainer not initialized")
        return self._explainer