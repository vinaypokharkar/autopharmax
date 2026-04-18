import sys
import os
from unittest.mock import MagicMock

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock ModelService and DataService modules BEFORE importing main
# This prevents the actual loading of heavy models/data during tests
mock_model_service_module = MagicMock()
mock_data_service_module = MagicMock()

# Setup ModelService mock
mock_model_instance = MagicMock()
mock_model_service_module.ModelService.return_value = mock_model_instance

# Setup DataService mock
mock_data_instance = MagicMock()
mock_data_service_module.DataService.return_value = mock_data_instance

# Inject mocks into sys.modules
sys.modules["services.model_service"] = mock_model_service_module
sys.modules["services.data_service"] = mock_data_service_module

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Drug Response Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/health",
            "options": "/api/v1/options",
            "predict": "/api/v1/predict"
        }
    }

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
