## 💊 AutoPharma — Drug Response Prediction

Predict cancer drug response (LN(IC50)) from GDSC cell-line and drug data. This repo includes data preparation notebooks, a full model training pipeline, and a Streamlit app for interactive inference.

---

### 📦 Repository Structure

- `src/`: Core ML logic (data ingestion, preprocessing, training).
  - `src/data/`: Data handling scripts.
  - `src/models/`: Training and evaluation scripts.
- `backend/`: FastAPI backend for serving the model.
- `data/`: Data storage.
  - `data/raw/`: Raw datasets.
  - `data/processed/`: Processed datasets.
- `models/`: Trained model artifacts (pickle files).
- `notebooks/`: Jupyter notebooks for experimentation.
- `tests/`: Unit and integration tests.
- `main.py`: Entry point for the application.
- `requirements.txt`: Python dependencies.

---

### 🐳 Running the Application with Docker

This project is containerized using Docker and Docker Compose.

#### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

#### Instructions

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd autopharma
    ```

2.  **Build and run the containers:**

    ```bash
    docker-compose up --build -d
    ```

3.  **Access the application:**
    - **Frontend:** [http://localhost:8501](http://localhost:8501)
    - **Backend API:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 📂 Data

Place raw files under `data/raw/`:

- `data/raw/GDSC_DATASET.csv`
- `data/raw/Cell_Lines_Details.xlsx`

Generate the merged dataset (if not already present):

1. Open `notebooks/mergedatasets.ipynb`
2. Run all cells — it will create `data/processed/merged_data.csv`

---

### 🧠 Training Pipeline

You can run the training pipeline using the script in `src/models/train.py`.

```bash
python src/models/train.py
```

Or explore via notebooks:

Use `notebooks/after.ipynb` to:

- Load `data/processed/merged_data.csv`
- Identify numeric and categorical columns
- Standardize numeric features
- Optionally one-hot encode categorical features and apply PCA
- Split data (Stratified by `TCGA_DESC`): Train 60%, Val 20%, Test 20%
- Train and evaluate: Linear Regression, ElasticNet, Random Forest, XGBoost
- Hyperparameter tune RF and XGB using validation set
- Select best model (by Val RMSE) and export artifacts to `models/`:
  - `best_model_XGBoost_(Tuned).pkl`
  - `model_XGBoost_Tuned.pkl` and other baselines
  - `scaler.pkl` used at inference
  - `model_metadata.pkl` containing `best_model_name`, `feature_columns`, `training_date`, and results summary

---

### 🛠️ Troubleshooting

- **Containers not starting:**
  - Check the logs of the containers for errors: `docker-compose logs <service-name>` (e.g., `docker-compose logs backend`).
  - Ensure that the required ports (8000 and 8501) are not already in use on your host machine.
- **Connection issues between frontend and backend:**
  - Make sure you are using the correct service name (`backend`) in the frontend code to connect to the backend API.

---

### 📄 License

For hackathon/demo purposes. Add a license if you plan to distribute.
