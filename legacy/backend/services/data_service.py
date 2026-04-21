import pandas as pd
from typing import Optional

class DataService:
    """
    Service for loading and managing data
    Singleton pattern ensures data is loaded only once
    """
    _instance = None
    _original_data = None
    
    def __new__(cls):
        """Singleton pattern - only one instance of DataService"""
        if cls._instance is None:
            cls._instance = super(DataService, cls).__new__(cls)
            cls._instance._load_data()
        return cls._instance
    
    def _load_data(self):
        """Load original data from CSV file"""
        # Use absolute path relative to this file to be robust
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(base_dir, "data", "processed", "merged_data.csv")

        
        try:
            self._original_data = pd.read_csv(data_path)
            print(f"✅ Data loaded successfully: {len(self._original_data)} rows")
            print(f"   Columns: {list(self._original_data.columns)}")
        except FileNotFoundError:
            print(f"❌ Error: File not found at {data_path}")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def get_original_data(self) -> pd.DataFrame:
        """
        Get the loaded original data
        
        Returns:
            pd.DataFrame: The original dataset
        
        Raises:
            ValueError: If data is not loaded
        """
        if self._original_data is None:
            raise ValueError("Data not loaded")
        return self._original_data
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics of the data
        
        Returns:
            dict: Summary information about the dataset
        """
        if self._original_data is None:
            raise ValueError("Data not loaded")
        
        return {
            "total_rows": len(self._original_data),
            "total_columns": len(self._original_data.columns),
            "cell_lines": self._original_data['TCGA_DESC'].nunique() if 'TCGA_DESC' in self._original_data.columns else 0,
            "drugs": self._original_data['DRUG_NAME'].nunique() if 'DRUG_NAME' in self._original_data.columns else 0,
            "columns": list(self._original_data.columns)
        }
    
    def get_cell_lines(self) -> list:
        """
        Get list of unique cell lines
        
        Returns:
            list: Sorted list of cell line names
        """
        if self._original_data is None:
            raise ValueError("Data not loaded")
        
        if 'TCGA_DESC' in self._original_data.columns:
            return sorted(self._original_data['TCGA_DESC'].unique().tolist())
        return []
    
    def get_drugs(self) -> list:
        """
        Get list of unique drugs
        
        Returns:
            list: Sorted list of drug names
        """
        if self._original_data is None:
            raise ValueError("Data not loaded")
        
        if 'DRUG_NAME' in self._original_data.columns:
            return sorted(self._original_data['DRUG_NAME'].unique().tolist())
        return []
    
    def filter_data(self, cell_line: str, drug_name: str) -> pd.DataFrame:
        """
        Filter data by cell line and drug name
        
        Args:
            cell_line: The cancer cell line name
            drug_name: The drug name
        
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        if self._original_data is None:
            raise ValueError("Data not loaded")
        
        filtered = self._original_data[
            (self._original_data['TCGA_DESC'] == cell_line) &
            (self._original_data['DRUG_NAME'] == drug_name)
        ]
        
        return filtered