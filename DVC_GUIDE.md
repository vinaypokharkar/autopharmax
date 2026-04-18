# Implementing Data Version Control (DVC)

This guide will help you set up DVC to track your large datasets and trained models, essentially giving you "git for data".

## Prerequisites

Ensure you are in the root directory of your project: `c:\Users\vinay\OneDrive\Desktop\hackathon\autopharma`

## Step 1: Installation

We added `dvc` to your `backend/requirement.txt`, so install it:

```bash
pip install -r backend/requirement.txt
# OR install it globally for better CLI usage
pip install dvc
```

## Step 2: Initialize DVC

Run this in your project root:

```bash
dvc init
```

This creates a `.dvc` directory. Move it to git:

```bash
git commit -m "Initialize DVC"
```

## Step 3: Track Your Data

We want to track the `dataset` folder.

1.  **Stop tracking with git (if tracked):**

    ```bash
    git rm -r --cached dataset
    git commit -m "Stop tracking dataset with git"
    ```

    _(If git wasn't checking it in, you can skip this, but it's safe to run)._

2.  **Add to DVC:**

    ```bash
    dvc add dataset
    ```

    This creates a file `dataset.dvc`. This file is small and safe for git.

3.  **Track trained models (Optional but recommended):**
    ```bash
    dvc add trained_models
    ```
    _(Assuming your models are in the root `trained_models` folder)._

## Step 4: Configure Remote Storage (The "Cloud")

To share data with teammates or your production server, you need a remote. For now, we'll simulate this with a local folder, but in production, you'd use S3/GCS.

```bash
# Create a folder outside your project to act as the "cloud"
mkdir c:\Users\vinay\OneDrive\Desktop\dvc_storage

# Configure DVC to use it
dvc remote add -d local_storage c:\Users\vinay\OneDrive\Desktop\dvc_storage
```

## Step 5: Save Changes

Now, commit the DVC tracking files to git:

```bash
git add dataset.dvc trained_models.dvc .gitignore
git commit -m "Track data and models with DVC"
```

## Step 6: Push Data

Whenever you change the dataset, run:

```bash
dvc add dataset
dvc push
```

## Workflow Summary

- **Code changes?** -> `git add .` -> `git commit` -> `git push`
- **Data changes?** -> `dvc add dataset` -> `git add dataset.dvc` -> `git commit` -> `dvc push`
