"""Utility helpers for multimodal price prediction."""
from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np
import requests
from PIL import Image
from importlib import import_module
from importlib import util as importlib_util

from .config import PATHS

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(log_level: int = logging.INFO) -> None:
    """Configure application-wide logging once."""

    PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(log_level)
        return
    logging.basicConfig(
        level=log_level,
        format=_LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(PATHS.logs_dir / "pipeline.log", mode="a"),
        ],
    )


def set_global_seed(seed: int = 42) -> None:
    """Ensure reproducible behavior across libraries."""

    random.seed(seed)
    np.random.seed(seed)
    if importlib_util.find_spec("torch") is None:
        logging.debug("Torch not available; skipping torch seed setup")
        return
    torch = import_module("torch")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Symmetric Mean Absolute Percentage Error in percent."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator == 0
    denominator[mask] = 1.0
    diff = np.abs(y_pred - y_true) / denominator
    diff[mask] = 0.0
    return float(np.mean(diff) * 100.0)


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


def ensure_project_structure() -> None:
    """Make sure commonly used folders are present."""

    for path in (
        PATHS.data_processed,
        PATHS.models_dir,
        PATHS.outputs,
        PATHS.logs_dir,
        PATHS.reports_dir,
    ):
        ensure_directory(path)


def save_json(data: Any, path: Path) -> None:
    """Persist Python data structures as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Load JSON content from disk."""

    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def download_image_with_retry(
    url: str,
    destination: Path,
    retries: int = 3,
    timeout: int = 10,
    backoff_factor: float = 1.5,
) -> Optional[Path]:
    """Download an image with retry logic and exponential backoff."""

    ensure_directory(destination.parent)
    session = requests.Session()
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            destination.write_bytes(response.content)
            return destination
        except (requests.RequestException, OSError) as exc:
            logging.warning("Image download failed (%s/%s): %s", attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(backoff_factor ** attempt)
    logging.error("Image download ultimately failed: %s", url)
    return None


def load_image(path: Path, size: Tuple[int, int] = (224, 224)) -> Optional[Image.Image]:
    """Load and resize an image if present."""

    try:
        image = Image.open(path).convert("RGB")
        if size:
            image = image.resize(size)
        return image
    except (OSError, FileNotFoundError) as exc:
        logging.warning("Failed to load image %s: %s", path, exc)
        return None


def chunk_iterable(sequence: Iterable[Any], chunk_size: int) -> Iterable[list[Any]]:
    """Yield successive chunked lists from an iterable."""

    chunk: list[Any] = []
    for item in sequence:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def cache_array(array: np.ndarray, path: Path) -> None:
    """Persist a NumPy array in compressed format."""

    ensure_directory(path.parent)
    np.savez_compressed(path, array=array)


def load_cached_array(path: Path) -> np.ndarray:
    """Load a cached NumPy array."""

    with np.load(path) as data:
        return data["array"]


def linear_weight_search(
    val_scores: dict[str, float],
    weight_grid: Iterable[Tuple[float, float, float]],
) -> Tuple[str, float]:
    """Find best weighted SMAPE score given validation metrics."""

    best_combo: Optional[Tuple[str, float]] = None
    for weights in weight_grid:
        w_gbm, w_transformer, w_nn = weights
        if not math.isclose(w_gbm + w_transformer + w_nn, 1.0, rel_tol=1e-6):
            continue
        score = (
            w_gbm * val_scores["gbm"]
            + w_transformer * val_scores["transformer"]
            + w_nn * val_scores["nn"]
        )
        if best_combo is None or score < best_combo[1]:
            best_combo = (f"{w_gbm:.2f}-{w_transformer:.2f}-{w_nn:.2f}", score)
    if best_combo is None:
        raise ValueError("No valid weight combination found.")
    return best_combo


def timeit(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to log runtime of long operations."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            logging.info("Starting %s", name)
            result = func(*args, **kwargs)
            duration = time.time() - start
            logging.info("Finished %s in %.2f seconds", name, duration)
            return result

        return wrapper

    return decorator
