# Implementing Continuous Training (CT)

This guide explains how to automatically retrain your model when new data is added.

## Concept: CI vs CT

- **CI (Continuous Integration):** We already set this up. It runs tests (`pytest`) to ensure code isn't broken. It does NOT retrain the model.
- **CT (Continuous Training):** This new pipeline will run `train.py` to update the model weights (`.pkl` file) when the dataset changes.

## New Files Created

1.  **`backend/train.py`**: A clean Python script extracted from your notebooks. It:
    - Loads `dataset/merged_data.csv`.
    - Retrains the XGBoost model.
    - Saves the new model to `trained_models/`.
    - Updates metadata and scaler.

2.  **`.github/workflows/retrain.yml`**: A GitHub Action that:
    - Triggers when `dataset/**` changes.
    - Installation dependencies.
    - Runs `python backend/train.py`.
    - Commits the updated `.pkl` model back to the repo.

## The Challenge: Data Access in GitHub

GitHub Actions runs on empty virtual machines. They do not have your local `dataset/` folder unless:

1.  **Option A (Bad Practice but Easy):** You commit the CSV to git (only checking if it's <100MB).
2.  **Option B (Best Practice):** You configure a DVC Remote (like S3 or Google Drive) and use `dvc pull` in the workflow.

## How to Test This

### 1. Manual Trigger

1.  Go to GitHub -> Actions -> "Continuous Training".
2.  Click "Run workflow".
3.  It will fail if `dataset/merged_data.csv` is not found (because it's gitignored).

### 2. Making it Work for Helper Demo

To make this work for a hackathon demo without setting up S3 buckets:

1.  **Commit a small sample dataset** to git: `dataset/merged_data.csv`.
2.  Push it.
3.  The action will trigger, retrain, and commit the new model.

## Recommendation

For professional MLOps:

1.  Use **CML (Continuous Machine Learning)** instead of raw Actions. CML lets you visualize the results (plots/metrics) directly in a Pull Request comment.
2.  Use **Iterative Studio** or **DAGsHub** to host your DVC storage for free.
