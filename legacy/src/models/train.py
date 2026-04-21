
import pandas as pd
import numpy as np
import pickle
import os
import sys
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.preprocess import load_and_preprocess_data

# Constants
# Relative to this script: ../../data/processed/merged_data.csv
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "merged_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model_XGBoost_(Tuned).pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.pkl")

def train():
    """Train XGBoost model and save artifacts."""
    # Create model directory if not exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print(f"Loading data from {DATA_PATH}...")
    try:
        # Load Data using src module
        df, feature_cols = load_and_preprocess_data(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    X = df[feature_cols]
    y = df['LN_IC50']
    
    print(f"Features: {len(feature_cols)}, Samples: {len(df)}")
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Scale Features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Model
    print("Training XGBoost model...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"✅ Training Complete. MSE: {mse:.4f}, R2: {r2:.4f}")
    
    # 6. Save Artifacts
    print("Saving model artifacts...")
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    # updated metadata
    metadata = {
        'feature_columns': feature_cols,
        'best_model_name': 'XGBoost (Retrained)',
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {'mse': mse, 'r2': r2}
    }
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
        
    print(f"Artifacts saved to {MODEL_DIR}")

    # Write metrics to a file for CML/GitHub Actions to read (in root)
    metrics_path = os.path.join(BASE_DIR, "metrics.txt")
    with open(metrics_path, "w") as outfile:
        outfile.write(f"R2_SCORE={r2:.4f}\n")
        outfile.write(f"MSE={mse:.4f}\n")

if __name__ == "__main__":
    train()
