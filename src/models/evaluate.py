
import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.preprocess import load_and_preprocess_data

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "merged_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model_XGBoost_(Tuned).pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.pkl")

def evaluate():
    print("Loading data for evaluation...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    try:
        # Using the same loading logic as training to ensure consistency
        df, feature_cols = load_and_preprocess_data(DATA_PATH)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Loading model and artifacts...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(METADATA_PATH):
        print(f"Error: Model artifacts not found in {MODEL_DIR}")
        return

    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return

    # Check features consistency
    saved_cols = metadata.get('feature_columns', [])
    if saved_cols and saved_cols != feature_cols:
        print("Warning: Feature columns in data differ from trained model metadata.")
        # Attempt to align columns
        common_cols = [c for c in saved_cols if c in df.columns]
        missing_cols = set(saved_cols) - set(df.columns)
        if missing_cols:
            print(f"Error: Missing features required by model: {missing_cols}")
            return
        feature_cols = saved_cols

    X = df[feature_cols]
    y_actual = df['LN_IC50']

    print(f"Model: {metadata.get('best_model_name', 'Unknown')}")
    print(f"Features: {len(feature_cols)}, Samples: {len(df)}")

    # Scale features
    print("Scaling features...")
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        print(f"Error scaling features: {e}")
        return

    # Predict
    print("Generating predictions...")
    try:
        y_pred = model.predict(X_scaled)
    except Exception as e:
        print(f"Error generating predictions: {e}")
        return

    # Calculate metrics
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)

    print("-" * 30)
    print(f"Performance Metrics (Full Dataset):")
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    print("-" * 30)

    # Plot
    print("Generating scatter plot...")
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    sns.scatterplot(x=y_actual, y=y_pred, alpha=0.5, edgecolor=None)
    
    # Add diagonal line (Perfect Prediction)
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.title(f'Actual vs Predicted LN(IC50) (XGBoost)\nR2: {r2:.3f}, MAE: {mae:.3f}')
    plt.xlabel('Actual LN(IC50)')
    plt.ylabel('Predicted LN(IC50)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_filename = 'xgboost_predicted_vs_actual.png'
    output_path = os.path.join(MODEL_DIR, output_filename) # Save in models/
        
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Scatter plot saved successfully to: {output_path}")

if __name__ == "__main__":
    evaluate()
