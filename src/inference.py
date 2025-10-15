"""Inference pipeline to generate competition submission."""
from __future__ import annotations

import logging
from pathlib import Path

from .config import PATHS, TRAINING
from .model_ensemble import build_ensemble, save_submission
from .utils import ensure_project_structure, setup_logging


def run_inference(feature_dir: Path | None = None, output_path: Path | None = None) -> Path:
    setup_logging()
    ensure_project_structure()
    result = build_ensemble(feature_dir)
    logging.info("Ensemble validation SMAPE: %.3f", result.validation_score)
    logging.info("Ensemble weights: %s", result.weights)
    return save_submission(result.test_predictions, output_path)


if __name__ == "__main__":  # pragma: no cover - manual execution
    submission_path = run_inference()
    logging.info("Submission generated at %s", submission_path)
