import argparse
import sys
import os

# Add project root to python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.pipelines.training_pipeline import run_training_pipeline
from src.data.merge import merge_datasets
from src.config.config import DATA_PATHS

def main():
    parser = argparse.ArgumentParser(description="ML Project Entry Point")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run the full training pipeline")

    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Run only the dataset merging step")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run the inference API")

    args = parser.parse_args()

    if args.command == "train":
        print("Running Training Pipeline...")
        run_training_pipeline()
    elif args.command == "merge":
        print("Running Dataset Merge...")
        merge_datasets(
            DATA_PATHS['gdsc_raw'],
            DATA_PATHS['cell_lines_raw'],
            DATA_PATHS['merged_data']
        )
    elif args.command == "serve":
        print("Starting Inference API...")
        # We import here to avoid loading backend dependencies if not needed
        import uvicorn
        # Assuming run is from root, backend package is importable
        uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
