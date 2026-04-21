"""
Orchestrates the training pipeline.
"""
import logging
import os
import sys

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)


from src.data.merge import merge_datasets
from src.models.train import train_tuned_model
from src.config.config import DATA_PATHS, MODEL_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training_pipeline():
    """
    Runs the full end-to-end training pipeline.
    """
    logger.info("Starting training pipeline...")
    
    # 1. Merge Datasets
    logger.info("Step 1: Merging Datasets")
    merge_datasets(
        DATA_PATHS['gdsc_raw'],
        DATA_PATHS['cell_lines_raw'],
        DATA_PATHS['merged_data']
    )
    
    # 3. Model Training & Evaluation (Combined in new function)
    logger.info("Step 3: Training & Evaluation")
        
    # Ensure output directory exists for models
    models_dir = os.path.dirname(MODEL_PATH)
    
    train_tuned_model(
        data_path=DATA_PATHS['merged_data'],
        output_dir=models_dir
    )
    
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    run_training_pipeline()
