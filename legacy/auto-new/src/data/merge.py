import pandas as pd
import os
import sys

def merge_datasets(gdsc_path: str, cell_lines_path: str, output_path: str):
    """
    Merges the GDSC drug response dataset with Cell Line details.
    
    Args:
        gdsc_path (str): Path to the GDSC CSV file.
        cell_lines_path (str): Path to the Cell Lines Excel file.
        output_path (str): Path where the merged CSV should be saved.
    """
    
    # 1. Validation
    if not os.path.exists(gdsc_path):
        print(f"❌ Error: GDSC dataset not found at: {gdsc_path}")
        return
    if not os.path.exists(cell_lines_path):
        print(f"❌ Error: Cell Lines dataset not found at: {cell_lines_path}")
        return

    print("⏳ Loading datasets...")
    try:
        # Load datasets
        gdsc = pd.read_csv(gdsc_path)
        # Note: Using openpyxl engine for .xlsx files
        cell_lines = pd.read_excel(cell_lines_path, engine='openpyxl')
    except Exception as e:
        print(f"❌ Error reading files: {e}")
        return

    print(f"   GDSC Shape: {gdsc.shape}")
    print(f"   Cell Lines Shape: {cell_lines.shape}")

    # 2. Merge Logic
    print("⏳ Merging datasets...")
    # Left join on COSMIC IDs to keep drug response data and add cell line metadata
    merged = gdsc.merge(
        cell_lines,
        left_on='COSMIC_ID',
        right_on='COSMIC identifier',
        how='left'
    )

    # 3. Data Cleaning
    initial_rows = len(merged)
    # Drop rows where any column has a missing value (NaN)
    merged = merged.dropna()
    final_rows = len(merged)
    
    dropped_rows = initial_rows - final_rows
    print(f"   Dropped {dropped_rows} rows containing missing values.")

    # 4. Save Output
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        merged.to_csv(output_path, index=False)
        print(f"✅ Success! Merged dataset saved to: {output_path}")
        print(f"   Final Shape: {merged.shape}")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")

if __name__ == "__main__":
    # Ensure project root is in path for imports to work
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from src.config.config import DATA_PATHS

    print(f"Using configuration from: {project_root}")
    
    merge_datasets(
        DATA_PATHS['gdsc_raw'],
        DATA_PATHS['cell_lines_raw'],
        DATA_PATHS['merged_data']
    )