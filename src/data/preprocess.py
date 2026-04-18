import pandas as pd
import numpy as np
import os

def load_and_preprocess_data(data_path: str):
    """
    Load and preprocess data from the given path.
    
    Args:
        data_path (str): Path to the CSV file.
        
    Returns:
        tuple: (pd.DataFrame, list) - The preprocessed dataframe and list of feature columns.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Simple preprocessing: Drop unused columns for training (ID columns) but keep features
    # Assuming 'LN_IC50' is target
    if 'LN_IC50' not in df.columns:
        raise ValueError("Target column 'LN_IC50' not found in dataset")
    
    # Identify numeric columns as features (excluding target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'LN_IC50' in numeric_cols:
        numeric_cols.remove('LN_IC50')
    
    # Drop rows with NaN
    df = df.dropna(subset=numeric_cols + ['LN_IC50'])
    
    return df, numeric_cols
