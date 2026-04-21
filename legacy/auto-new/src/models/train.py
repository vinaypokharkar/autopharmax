
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import pickle
import sys
import yaml
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# CONFIGURATION
# ==========================================
# Update this path to point to your merged CSV file
DEFAULT_DATA_PATH = "data/processed/merged_data.csv"
MODEL_OUTPUT_DIR = "models"
PARAMS_PATH = "params.yaml"

def load_params(params_path=PARAMS_PATH):
    """Load parameters from yaml file"""
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
            return params.get('train', {})
    except Exception as e:
        print(f"⚠️ Warning: Could not load params.yaml: {e}. Using defaults.")
        return {}

def train_tuned_model(data_path=DEFAULT_DATA_PATH, output_dir=MODEL_OUTPUT_DIR):
    """
    Trains a model using parameters from params.yaml and tracks experiment with MLflow.
    Support for XGBoost and RandomForest.
    """
    
    # 0. Load Params
    params = load_params()
    print(f"⚙️ Loaded params: {params}")

    model_type = params.get("model_type", "xgboost").lower()
    
    # MLflow Setup
    mlflow.set_experiment("Model-Training")

    with mlflow.start_run():
        # Log all params from params.yaml
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_type)

        # 1. Load Data
        if not os.path.exists(data_path):
            print(f"❌ Error: Dataset not found at {data_path}")
            return

        print(f"⏳ Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # 2. Preprocessing
        # Target variable
        target_col = 'LN_IC50'
        
        if target_col not in df.columns:
            print(f"❌ Error: Target column '{target_col}' not found.")
            return

        # Feature Selection: Use only Numeric Columns
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_exclude = [target_col] 
        feature_cols = [c for c in numeric_features if c not in cols_to_exclude]
        
        print(f"   Selected {len(feature_cols)} numeric features for training.")
        mlflow.log_param("num_features", len(feature_cols))

        X = df[feature_cols]
        y = df[target_col]
        
        # Remove rows with NaN
        X = X.dropna()
        y = y[X.index] # Align y with X
        
        # 3. Split Data
        print("⏳ Splitting data (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 4. Scaling
        print("⏳ Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 5. Train Model
        print(f"🚀 Training {model_type}...")
        
        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=params.get('n_estimators', 200),
                max_depth=params.get('max_depth', None),
                random_state=params.get('random_state', 42),
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            mlflow.sklearn.log_model(model, "model")
            
        elif model_type == "xgboost":
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=params.get('n_estimators', 200),
                max_depth=params.get('max_depth', 9),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 1.0),
                colsample_bytree=params.get('colsample_bytree', 1.0),
                random_state=params.get('random_state', 42),
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            mlflow.xgboost.log_model(model, "model")
        
        else:
            print(f"❌ Error: Unsupported model type '{model_type}'")
            return
        
        # 6. Evaluate
        print("⏳ Evaluating...")
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_test = r2_score(y_test, y_pred_test)
        
        # Log Metrics
        mlflow.log_metric("train_rmse", rmse_train)
        mlflow.log_metric("test_rmse", rmse_test)
        mlflow.log_metric("test_r2", r2_test)
        
        print("-" * 40)
        print(f"✅ Training Complete ({model_type})!")
        print(f"   Train RMSE: {rmse_train:.4f}")
        print(f"   Test RMSE:  {rmse_test:.4f}")
        print(f"   Test R²:    {r2_test:.4f}")
        print("-" * 40)
        
        # 7. Save Backend Artifacts (Legacy Support)
        os.makedirs(output_dir, exist_ok=True)
        
        model_filename = f"best_model_{model_type}.pkl"
        # Always output to specific file for DVC tracking if strictly needed, 
        # or just save as general model file. 
        # For DVC pipeline consistency currently we target 'best_model_XGBoost_(Tuned).pkl' 
        # We might want to standardize this name or update dvc.yaml.
        # Strict adherence to dvc.yaml outs:
        # Strictly enforce the output name defined in dvc.yaml outs
        # This ensures dvc repro doesn't complain about missing outputs
        model_path = os.path.join(output_dir, "best_model_XGBoost_(Tuned).pkl")

        scaler_path = os.path.join(output_dir, "scaler.pkl")
        metadata_path = os.path.join(output_dir, "model_metadata.pkl")
        
        print(f"💾 Saving artifacts to '{output_dir}/'...")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Log artifacts to MLflow
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)

        # Valid metadata for reproducibility
        metadata = {
            'feature_columns': feature_cols,
            'best_model_name': model_type,
            'params': model.get_params(),
            'metrics': {'rmse': rmse_test, 'r2': r2_test}
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        mlflow.log_artifact(metadata_path)
            
        print("✅ All files saved and tracked in MLflow.")

if __name__ == "__main__":
    train_tuned_model()