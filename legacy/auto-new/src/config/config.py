"""
Configuration module for paths and parameters.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATHS = {
    "raw": os.path.join(PROJECT_ROOT, "data", "raw"),
    "processed": os.path.join(PROJECT_ROOT, "data", "processed"),
    "gdsc_raw": os.path.join(PROJECT_ROOT, "data", "raw", "GDSC2-dataset.csv"),
    "cell_lines_raw": os.path.join(PROJECT_ROOT, "data", "raw", "Cell_Lines_Details.xlsx"),
    "merged_data": os.path.join(PROJECT_ROOT, "data", "processed", "merged_data.csv")
}

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_model.json")

MODEL_CONFIG = {
    "learning_rate": 0.1,
    "max_depth": 5,
    "n_estimators": 100
}
