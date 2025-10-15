"""Data loading, cleaning, and exploratory analysis utilities."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import PATHS, TRAINING
from .utils import ensure_directory, ensure_project_structure, save_json, set_global_seed, setup_logging, timeit

_TEXT_COLUMNS = ["catalog_content"]
_IMAGE_COLUMN = "image_link"


@dataclass
class DataArtifacts:
    """Container for cached preprocessing outputs."""

    cleaned_data_path: Path
    exploratory_report_path: Path
    price_distribution_path: Path
    missing_summary_path: Path


def _log_basic_info(df: pd.DataFrame, label: str) -> None:
    logging.info("%s shape: %s", label, df.shape)
    logging.info("%s memory usage: %.2f MB", label, df.memory_usage(deep=True).sum() / 1e6)


def _price_bins(prices: pd.Series, bins: int = TRAINING.stratify_bins) -> pd.Series:
    quantiles = np.linspace(0, 1, bins + 1)
    labels = range(bins)
    return pd.qcut(prices, q=quantiles, labels=labels, duplicates="drop")


def read_csv_chunked(path: Path, chunk_size: int = 50_000) -> pd.DataFrame:
    """Load large CSV files without exhausting memory."""

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


@timeit("load-datasets")
def load_datasets(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read training and test datasets."""

    set_global_seed(TRAINING.random_seed)
    train_df = read_csv_chunked(train_path)
    test_df = read_csv_chunked(test_path)
    _log_basic_info(train_df, "train")
    _log_basic_info(test_df, "test")
    return train_df, test_df


def analyze_missing_values(df: pd.DataFrame, label: str, output_path: Path) -> Dict[str, float]:
    """Compute missing value ratios and persist them."""

    missing_share = (df.isna().mean() * 100).sort_values(ascending=False)
    summary = missing_share.to_dict()
    save_json(summary, output_path)
    logging.info("Missing value summary for %s saved to %s", label, output_path)
    return summary


def analyze_text_statistics(df: pd.DataFrame, output_path: Path) -> Dict[str, float]:
    """Extract descriptive statistics for text columns."""

    stats: Dict[str, float] = {}
    for column in _TEXT_COLUMNS:
        text_lengths = df[column].fillna("").astype(str).apply(len)
        stats[f"{column}_char_mean"] = float(text_lengths.mean())
        stats[f"{column}_char_std"] = float(text_lengths.std())
        stats[f"{column}_char_p95"] = float(text_lengths.quantile(0.95))
        word_lengths = df[column].fillna("").astype(str).str.split().apply(len)
        stats[f"{column}_word_mean"] = float(word_lengths.mean())
        stats[f"{column}_word_std"] = float(word_lengths.std())
        stats[f"{column}_word_p95"] = float(word_lengths.quantile(0.95))
    save_json(stats, output_path)
    logging.info("Text statistics saved to %s", output_path)
    return stats


def price_outlier_detection(df: pd.DataFrame) -> Dict[str, float]:
    """Detect price outliers using IQR and Z-score heuristics."""

    prices = df[TRAINING.target_column].astype(float)
    q1 = prices.quantile(0.25)
    q3 = prices.quantile(0.75)
    iqr = q3 - q1
    iqr_bounds = (max(0.0, q1 - 1.5 * iqr), q3 + 1.5 * iqr)
    z_scores = np.abs((prices - prices.mean()) / prices.std(ddof=0))
    z_score_outliers = float((z_scores > 3).mean() * 100)
    return {
        "iqr_lower": float(iqr_bounds[0]),
        "iqr_upper": float(iqr_bounds[1]),
        "z_score_outlier_pct": z_score_outliers,
        "median_price": float(prices.median()),
        "mean_price": float(prices.mean()),
        "min_price": float(prices.min()),
        "max_price": float(prices.max()),
    }


def category_price_ranges(df: pd.DataFrame, category_column: str = "brand") -> Dict[str, Dict[str, float]]:
    """Compute price summary stats per category/brand."""

    if category_column not in df.columns:
        logging.warning("Category column %s missing; skipping price range analysis", category_column)
        return {}
    grouped = df.groupby(category_column)[TRAINING.target_column]
    summary = {
        str(cat): {
            "median": float(vals.median()),
            "mean": float(vals.mean()),
            "count": int(vals.count()),
        }
        for cat, vals in grouped
    }
    return summary


def fill_missing_text(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing text columns with empty strings."""

    for column in _TEXT_COLUMNS:
        df[column] = df[column].fillna("").astype(str)
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning to the dataset."""

    df = fill_missing_text(df)
    df[_IMAGE_COLUMN] = df[_IMAGE_COLUMN].fillna("")
    return df


def generate_bins_for_cv(train_df: pd.DataFrame) -> pd.Series:
    """Create stratification bins for cross-validation."""

    return _price_bins(train_df[TRAINING.target_column])


@timeit("preprocess-and-cache")
def preprocess_and_cache(train_df: pd.DataFrame, test_df: pd.DataFrame) -> DataArtifacts:
    """Run preprocessing steps and save intermediate artifacts."""

    ensure_directory(PATHS.data_processed)
    train_clean = preprocess_dataframe(train_df.copy())
    test_clean = preprocess_dataframe(test_df.copy())
    bins = generate_bins_for_cv(train_clean)
    train_clean["price_bin"] = bins
    train_path = PATHS.data_processed / "train_clean.parquet"
    test_path = PATHS.data_processed / "test_clean.parquet"
    train_clean.to_parquet(train_path, index=False)
    test_clean.to_parquet(test_path, index=False)

    artifacts = DataArtifacts(
        cleaned_data_path=train_path,
        exploratory_report_path=PATHS.reports_dir / "eda_summary.json",
        price_distribution_path=PATHS.reports_dir / "price_distribution.json",
        missing_summary_path=PATHS.reports_dir / "missing_summary.json",
    )

    ensure_directory(artifacts.exploratory_report_path.parent)
    price_stats = price_outlier_detection(train_clean)
    save_json(price_stats, artifacts.price_distribution_path)
    text_stats = analyze_text_statistics(train_clean, artifacts.exploratory_report_path)
    missing_stats = analyze_missing_values(train_clean, "train", artifacts.missing_summary_path)
    logging.info("Price stats: %s", price_stats)
    logging.info("Text stats keys: %s", list(text_stats))
    logging.info("Missing stats keys: %s", list(missing_stats))
    return artifacts


def compute_fold_smape(scores: Iterable[float]) -> float:
    """Aggregate SMAPE across validation folds."""

    scores = list(scores)
    return float(np.mean(scores))


def main(train_path: Path, test_path: Path) -> None:
    """Execute preprocessing routine."""

    setup_logging()
    ensure_project_structure()
    train_df, test_df = load_datasets(train_path, test_path)
    preprocess_and_cache(train_df, test_df)


if __name__ == "__main__":
    default_train = PATHS.locate_dataset_file("train.csv")
    default_test = PATHS.locate_dataset_file("test.csv")
    if default_train is not None and default_test is not None:
        main(default_train, default_test)
    else:
        logging.error(
            "Unable to locate train/test CSV files. Checked %s and %s",
            PATHS.data_raw,
            PATHS.student_dataset,
        )
