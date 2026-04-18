# Implementing CI/CD with GitHub Actions

This guide explains the automated testing pipeline we set up for your AutoPharma project.

## Overview

We created a GitHub Actions workflow that:

1.  Triggers on every `push` or `pull_request` to `main` or `master`.
2.  Sets up a Python 3.9 environment.
3.  Installs dependencies from `backend/requirement.txt`.
4.  Runs unit tests using `pytest`.

## Files Created

### 1. Workflow File (`.github/workflows/mlops.yml`)

This is the configuration file for GitHub Actions. It defines the steps to build and test your application.

```yaml
name: AutoPharma MLOps Pipeline
...
- name: Run Tests
  run: |
    cd backend
    export PYTHONPATH=$PYTHONPATH:.
    pytest tests/
```

### 2. Test File (`backend/tests/test_main.py`)

We created a test suite that checks your API's health without needing the heavy ML models loaded.

**Key Feature - Mocking:**
The test mocks `ModelService` and `DataService`. This means:

- The test runs fast.
- The test runs even if `trained_models/` folder is missing (which happens in CI if DVC isn't configured yet).
- The test verifies the API structure and response formats.

## How to Verify

### 1. Locally

Run the following command in your terminal:

```bash
# Ensure you have pytest installed
pip install pytest httpx

# Run tests
pytest backend/tests/test_main.py
```

You should see green passing tests.

### 2. On GitHub

1.  Push your changes to GitHub:
    ```bash
    git add .
    git commit -m "Add CI/CD pipeline and tests"
    git push origin main
    ```
2.  Go to your repository on GitHub.
3.  Click the **Actions** tab.
4.  You will see a workflow run named "AutoPharma MLOps Pipeline".
5.  If it's green, your CI/CD is working!

## Next Steps (Phase 3)

In the future, you can expand this pipeline to:

- **Linting:** Add `flake8` to check code style.
- **DVC Integration:** Pull real data/models to run integration tests.
- **Docker Build:** Automatically build and push your Docker image if tests pass.
