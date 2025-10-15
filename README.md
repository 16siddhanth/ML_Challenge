# Multimodal E-Commerce Price Prediction

This repository trains, evaluates, and serves a multimodal stacked ensemble that predicts product prices from catalog text, structured attributes, and optionally cached product images.

## Project Structure

```
.
├── data/
│   ├── raw/                # Place train.csv, test.csv here
│   └── processed/          # Cached parquet files and engineered features
├── logs/                   # Pipeline logs
├── models/                 # Trained model weights and metrics
├── notebooks/              # Exploratory notebooks (optional)
├── outputs/                # Submission files (test_out.csv)
├── reports/                # EDA summaries and diagnostics
└── src/                    # Source code package
```

## Quickstart

1. **Create environment & install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Place data** (challenge `train.csv`, `test.csv`) inside `data/raw/`. The pipeline also auto-detects the provided `student_resource/dataset/` folder if left unchanged.

3. **Run preprocessing + feature engineering**
   ```bash
   python -m src.data_preprocessing
   python -m src.feature_engineering              # add --image-embeddings if images are cached locally
   ```

4. **Tune & train models**
   ```bash
   python -m src.model_training --optuna-trials 75 --skip-train   # optional LightGBM tuning
   python -m src.model_training                                   # trains base models + meta stack
   ```

5. **Blend predictions & generate submission**
   ```bash
   python -m src.inference
   ```

The final submission will be saved to `outputs/test_out.csv`.

## Model Overview

- **LightGBM Tweedie** on a sparse matrix combining one-hot categories, scaled numeric features (IPQ, numeric specs), DistilBERT text embeddings, and optional ViT image embeddings.
- **CatBoost** consuming raw `catalog_content` as text features plus DistilBERT/ViT vectors via CatBoost embedding features.
- **XGBoost** trained with `reg:absoluteerror`, monotonic constraints on clearly monotone fields, and log-target handling.
- **Meta-stacker**: StandardScaler + HuberRegressor fitted on out-of-fold predictions to minimize SMAPE and enforce positive predictions.

## Running in hosted notebooks (Colab / Lightning AI)

1. Clone or upload the project, then `cd` into the workspace directory.
2. Install dependencies: `pip install -r requirements.txt`.
3. (Optional) Upload pre-downloaded images into `data/raw/images` and call `python -m src.feature_engineering --image-embeddings`. Without cached images, omit the flag and the pipeline will skip vision features.
4. Run the command sequence from the Quickstart. GPU, if available, is automatically used for embedding extraction; all other steps operate on CPU.

## Reproducibility

- Random seed fixed at 42 for NumPy, Python, LightGBM, and PyTorch.
- Intermediate features cached under `data/processed/` to avoid recomputation.
- Cross-validation uses stratified folds on price bins to stabilize SMAPE.

## Packaging for Submission

Challenge submission typically requires `outputs/test_out.csv` plus a one-page methodology summary (see `documentation.md`).
