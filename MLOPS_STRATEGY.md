# AutoPharma MLOps Pipeline Strategy

This document outlines the MLOps architecture designed to professionalize the lifecycle of the AutoPharma project. This pipeline demonstrates to professors that the project involves continuous integration, delivery, and training (CI/CD/CT).

```mermaid
flowchart LR
    subgraph Development [Data Science & Dev]
        Data[(Dataset)] -->|Version Control| DVC[DVC]
        Code[Source Code] -->|Push| Git[GitHub Repo]

        subgraph Experimentation
            TrainScript[train.py] -->|Log Metrics| MLflow[MLflow Tracking]
            TrainScript -->|Log Model| Registry[Model Registry]
        end
    end

    subgraph CI_CD [CI/CD Pipeline (GitHub Actions)]
        Git -->|Trigger| Action_Test[Unit Tests]
        Action_Test -->|Success| Action_Build[Build Docker Image]
        Action_Build -->|Push| DockerHub[Container Registry]
    end

    subgraph Production [Deployment]
        DockerHub -->|Pull| Server[Production Server]
        Server -->|Serve| API[FastAPI Backend]

        API -->|Logs| Monitoring[Drift Detection]
    end

    DVC -.-> TrainScript
    Registry -.-> API
```

## The 4 Pillars of Your MLOps Implementation

To impress your professors, you don't need to implement Google's entire infrastructure. You need these 4 components working together:

### 1. Experiment Tracking (The "Lab Notebook")

**Tool:** `MLflow`

- **Problem:** "I ran the model 5 times with different parameters, I forgot which one was best."
- **Solution:** Every time you run `train.py`, it automatically logs:
  - **Parameters:** (Learning rate, estimators, depth)
  - **Metrics:** (RMSE, R2, Accuracy)
  - **Artifacts:** (The saved `.pkl` model, plots)
- **Demo:** Show the MLflow UI dashboard with charts comparing different runs.

### 2. Data Version Control (The "Time Machine")

**Tool:** `DVC (Data Version Control)`

- **Problem:** "I changed the dataset yesterday and now the model is worse. I can't go back."
- **Solution:** DVC tracks large CSV files (which git can't handle).
- **Implementation:**
  - `dvc init`
  - `dvc add dataset/merged_data.csv`
  - This creates a small `.dvc` pointer file that you commit to Git.

### 3. CI/CD (The "Automation")

**Tool:** `GitHub Actions`

- **Problem:** "It works on my laptop but breaks on the server."
- **Solution:** Automated checks.
- **Implementation:** Create a workflow `.github/workflows/mlops.yml` that:
  1.  Sets up Python.
  2.  Installs requirements.
  3.  Runs a basic test (e.g., checks if model loads, checks input format).

### 4. Model Monitoring (The "Health Check")

**Tool:** `Evidently AI` (Optional but impressive)

- **Problem:** "The world changed, and my model is now outdated (Drift)."
- **Solution:** A script that compares "Training Data" stats vs "Live Data" stats.

---

## Your Implementation Plan (Step-by-Step)

### Phase 1: Tracking (Do this now)

1.  Install `mlflow`.
2.  Update your training script to use `mlflow.log_param()` and `mlflow.log_metric()`.
3.  Run the training script 3-4 times with different params.
4.  **Result:** You have a dashboard showing rigorous experimentation.

### Phase 2: Automation

1.  Create a simple test file `tests/test_api.py`.
2.  Create a GitHub Action config.
3.  **Result:** You have a green "Passing" badge on your repo.

### Phase 3: Versioning

1.  Initialize DVC.
2.  **Result:** You can prove you track data lineage.
