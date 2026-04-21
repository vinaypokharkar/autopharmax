# Auto-New ML Project

A clean, modular, and scalable MLOps-ready repository.

## Project Structure

- `data/`: Contains raw and processed data.
- `src/`: Core logic for data processing, model training, and pipelines.
- `backend/`: FastAPI application for model inference.
- `models/`: Stores trained model artifacts.
- `notebooks/`: Jupyter notebooks for experimentation.
- `tests/`: Unit and integration tests.
- `src/config/`: Configuration settings.

## Setup

1. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run Training Pipeline

```bash
python main.py train
```

### Run Inference API

```bash
python main.py serve
```

Then visit `http://localhost:8000/docs` for the API documentation.

## Architecture

- **Decoupling**: Training logic (`src/`) is separated from the backend API (`backend/`).
- **Config**: Centralized configuration in `src/config/config.py`.
- **Pipelines**: Orchestrated workflows in `src/pipelines/`.
