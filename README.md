## 💊 AutoPharma — Drug Response Prediction

Predict cancer drug response (LN(IC50)) from GDSC cell-line and drug data. This repo includes data preparation notebooks, a full model training pipeline, and a Streamlit app for interactive inference.

---

### 📦 Repository Structure

- `app.py`: Streamlit UI for predictions using the best saved model
- `after.ipynb`: EDA + feature engineering experiments (scaling, encoding, PCA, modeling)
- `mergedatasets.ipynb`: Merges raw GDSC and cell-line metadata into `dataset/merged_data.csv`
- `dataset/`
  - `GDSC_DATASET.csv`
  - `Cell_Lines_Details.xlsx`
  - `merged_data.csv`
- `trained_models/` (ignored by Git)
  - `best_model_XGBoost_(Tuned).pkl`
  - `model_*` variants and `scaler.pkl`
  - `model_metadata.pkl`
- `.gitignore`: excludes `trained_models/`

---

### 🖥️ Requirements

Python 3.10 is used in notebooks; Python 3.9–3.11 should work.

Install the core libraries:

```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost streamlit seaborn matplotlib openpyxl scipy
```

Optional (for faster tree models):

```bash
pip install lightgbm
```

---

### 📂 Data

Place raw files under `dataset/`:

- `dataset/GDSC_DATASET.csv`
- `dataset/Cell_Lines_Details.xlsx`

Generate the merged dataset (if not already present):

1) Open `mergedatasets.ipynb`
2) Run all cells — it will create `dataset/merged_data.csv`

Notes:
- The merged dataset is ~37MB; ensure paths are correct on Windows.
- Columns of interest include: `TCGA_DESC`, `DRUG_NAME`, `LN_IC50`, numerical features, and cell-line metadata.

---

### 🧠 Training Pipeline (reproducible via notebook)

Use `after.ipynb` to:

- Load `dataset/merged_data.csv`
- Identify numeric and categorical columns
- Standardize numeric features
- Optionally one-hot encode categorical features and apply PCA
- Split data (Stratified by `TCGA_DESC`): Train 60%, Val 20%, Test 20%
- Train and evaluate: Linear Regression, ElasticNet, Random Forest, XGBoost
- Hyperparameter tune RF and XGB using validation set
- Select best model (by Val RMSE) and export artifacts to `trained_models/`:
  - `best_model_XGBoost_(Tuned).pkl`
  - `model_XGBoost_Tuned.pkl` and other baselines
  - `scaler.pkl` used at inference
  - `model_metadata.pkl` containing `best_model_name`, `feature_columns`, `training_date`, and results summary

Paths are hard-coded for Windows in the notebook and app. Update if your user profile differs.

---

### 🚀 Run the App

The app expects the following files to exist:

- `dataset/merged_data.csv`
- `trained_models/best_model_XGBoost_(Tuned).pkl`
- `trained_models/scaler.pkl`
- `trained_models/model_metadata.pkl`

Start the Streamlit server from the repo root:

```bash
streamlit run app.py
```

In the UI:

- Select a cancer cell line (`TCGA_DESC`) and a drug (`DRUG_NAME`)
- Click “Predict IC50”
- The app builds the feature vector using `metadata['feature_columns']`, scales with `scaler.pkl`, predicts with the best model, and displays both LN(IC50) and IC50 (µM)
- If no matching row exists for the chosen pair, the app falls back to mean feature values and warns you

---

### ⚙️ Configuration & Paths

Default Windows paths used in code:

```text
dataset path:      C:\Users\vinay\OneDrive\Desktop\hackathon\autopharma\dataset\merged_data.csv
models directory:  C:\Users\vinay\OneDrive\Desktop\hackathon\autopharma\trained_models
```

If your local setup differs, update these in:

- `app.py` (functions `load_models()` and `load_original_data()`)
- Notebooks where files are read/written

---

### 🔒 Git Hygiene

`trained_models/` is ignored via `.gitignore`:

```text
trained_models/
```

This prevents large binaries (e.g., `model_Random_Forest_Tuned.pkl`, ~807MB) from entering your repo.

---

### 🧪 Reproducibility Tips

- Fix seeds where possible (`random_state=42`) for splits and models
- Save `model_metadata.pkl` after each training session to capture `feature_columns`
- Keep the same scaler used during training for inference

---

### 🛠️ Troubleshooting

- “Failed to load models or data” in the app:
  - Verify files exist in `trained_models/` and `dataset/`
  - Check hard-coded paths in `app.py`

- Different Windows username or folder layout:
  - Update absolute paths in code or switch to project-relative paths

- Memory/time issues during training:
  - Reduce features (e.g., fewer one-hot levels, or apply PCA)
  - Use smaller hyperparameter grids

---

### 📄 License

For hackathon/demo purposes. Add a license if you plan to distribute.


