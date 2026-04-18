
import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure we can import from 'services'
# This assumes the script is run from the 'backend' directory
# If run from root, we need to add 'backend' to sys.path
current_dir = os.getcwd()
if os.path.basename(current_dir) == 'autopharma':
    sys.path.append(os.path.join(current_dir, 'backend'))
    from backend.services.model_service import ModelService
    from backend.services.data_service import DataService
elif os.path.basename(current_dir) == 'backend':
    sys.path.append(current_dir)
    from services.model_service import ModelService
    from services.data_service import DataService
else:
    # Fallback: try to find the 'backend' directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    try:
        from services.model_service import ModelService
        from services.data_service import DataService
    except ImportError:
        print("Error: Could not import services. Please run this script from the project root or backend directory.")
        sys.exit(1)

def main():
    print("Initializing services...")
    try:
        # Initialize services
        # Note: DataService expects to be run relative to dataset/ folder structure
        # If run from backend/, it looks for ../dataset/merged_data.csv
        data_service = DataService()
        model_service = ModelService()
    except Exception as e:
        print(f"Error initializing services: {e}")
        print("Please ensure you are running this script from the correct directory (e.g., 'backend/').")
        return

    print("Loading data and model...")
    try:
        original_data = data_service.get_original_data()
        model = model_service.get_model()
        scaler = model_service.get_scaler()
        metadata = model_service.get_metadata()
    except Exception as e:
        print(f"Error loading resources: {e}")
        return

    feature_columns = metadata['feature_columns']
    
    print(f"Model: {metadata.get('best_model_name', 'Unknown')}")
    print(f"Number of Features: {len(feature_columns)}")
    
    # Check if 'LN_IC50' exists (Actual values)
    if 'LN_IC50' not in original_data.columns:
        print("Error: 'LN_IC50' column not found in dataset. Cannot compare Actual vs Predicted.")
        return

    print("Preparing data for prediction...")
    
    # Select feature columns + target
    try:
        data_to_use = original_data[feature_columns + ['LN_IC50']].copy()
    except KeyError as e:
        print(f"Error: Missing columns in dataset: {e}")
        return

    # Drop rows with NaN values in features or target
    initial_len = len(data_to_use)
    data_to_use = data_to_use.dropna()
    final_len = len(data_to_use)
    
    if initial_len != final_len:
        print(f"Dropped {initial_len - final_len} rows with missing values.")
        
    if final_len == 0:
        print("Error: No data left after dropping NaNs.")
        return

    X = data_to_use[feature_columns]
    y_actual = data_to_use['LN_IC50']
    
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
    print(f"Performance Metrics:")
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
    # Save to current directory or backend directory
    output_path = os.path.join(os.getcwd(), output_filename)
        
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Scatter plot saved successfully to: {output_path}")

if __name__ == "__main__":
    main()
