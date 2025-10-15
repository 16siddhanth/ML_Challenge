"""Model ensembling utilities."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import PATHS, TRAINING
from .model_training import DenseEnsembleNetwork
from .utils import ensure_directory, ensure_project_structure, setup_logging, smape, timeit


@dataclass
class EnsembleResult:
    """Container holding ensemble predictions and weights."""

    validation_predictions: np.ndarray
    test_predictions: np.ndarray
    weights: Dict[str, float]
    validation_score: float


def _load_sparse_matrix(path: Path):
    sparse_module = import_module("scipy.sparse")
    return sparse_module.load_npz(path)


def _prepare_sparse_features(cache_dir: Path) -> any:
    ohe = _load_sparse_matrix(cache_dir / "category_test.npz")
    numeric = np.load(cache_dir / "numeric_test.npy")
    sparse_module = import_module("scipy.sparse")
    components = [ohe, sparse_module.csr_matrix(numeric)]

    text_path = cache_dir / "text_embeddings_test.npz"
    if text_path.exists():
        with np.load(text_path) as data:
            text_embeddings = data["array"]
        components.append(sparse_module.csr_matrix(text_embeddings))

    image_path = cache_dir / "image_embeddings_test.npz"
    if image_path.exists():
        with np.load(image_path) as data:
            image_embeddings = data["array"]
        components.append(sparse_module.csr_matrix(image_embeddings))

    if len(components) == 1:
        return components[0].tocsr()
    return sparse_module.hstack(components, format="csr")


def _prepare_dense_features(cache_dir: Path) -> Optional[np.ndarray]:
    text_path = cache_dir / "text_embeddings_test.npz"
    image_path = cache_dir / "image_embeddings_test.npz"
    components: List[np.ndarray] = []
    if text_path.exists():
        with np.load(text_path) as data:
            components.append(data["array"])
    if image_path.exists():
        with np.load(image_path) as data:
            components.append(data["array"])
    if not components:
        return None
    return np.hstack(components)


def _load_structured_features(cache_dir: Path, split: str) -> Optional[pd.DataFrame]:
    path = cache_dir / f"structured_features_{split}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "sample_id" in df.columns:
        sample_ids = df["sample_id"].to_numpy()
        df = df.drop(columns=["sample_id"])
        if split == "test":
            try:
                test_ids = pd.read_parquet(PATHS.data_processed / "test_clean.parquet")["sample_id"].to_numpy()
                df = pd.DataFrame(df.to_numpy(), columns=df.columns, index=sample_ids)
                df = df.reindex(test_ids)
                df = df.reset_index(drop=True)
                df.columns = df.columns.astype(str)
                return df
            except FileNotFoundError:
                pass
    df.columns = df.columns.astype(str)
    return df


def _load_structured_metadata(cache_dir: Path) -> Dict[str, Iterable[str]]:
    meta_path = cache_dir / "structured_metadata.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return {
        "structured_columns": data.get("structured_columns", []),
        "categorical": data.get("structured_categorical", []),
    }


def _load_oof_predictions(model_name: str) -> Optional[np.ndarray]:
    path = PATHS.models_dir / f"oof_{model_name}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return data["array"]


def _load_gbm_models() -> List[Path]:
    gbm_dir = PATHS.models_dir / "gbm"
    if not gbm_dir.exists():
        return []
    return sorted(gbm_dir.glob("lightgbm_fold*.txt"))


def _predict_gbm(model_paths: Iterable[Path], X_sparse) -> np.ndarray:
    model_paths = list(model_paths)
    if not model_paths:
        return np.zeros(X_sparse.shape[0], dtype=np.float32)
    lgbm = import_module("lightgbm")
    preds = np.zeros(X_sparse.shape[0], dtype=np.float32)
    for path in model_paths:
        booster = lgbm.Booster(model_file=str(path))
        fold_preds = booster.predict(X_sparse, num_iteration=booster.best_iteration)
        if TRAINING.log_target:
            fold_preds = np.expm1(fold_preds)
        preds += fold_preds.astype(np.float32)
    preds /= max(len(model_paths), 1)
    return preds


def _load_catboost_models() -> List[Path]:
    model_dir = PATHS.models_dir / "catboost"
    if not model_dir.exists():
        return []
    return sorted(model_dir.glob("catboost_fold*.cbm"))


def _predict_catboost(model_paths: Iterable[Path], structured_features: Optional[pd.DataFrame], categorical_columns: Iterable[str]) -> np.ndarray:
    model_paths = list(model_paths)
    if not model_paths or structured_features is None or structured_features.empty:
        return np.zeros(0, dtype=np.float32)
    try:
        catboost_mod = import_module("catboost")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        logging.warning("CatBoost not available for inference: %s", exc)
        return np.zeros(0, dtype=np.float32)

    feature_df = structured_features.copy()
    cat_columns = [col for col in categorical_columns if col in feature_df.columns]
    pool = catboost_mod.Pool(feature_df, cat_features=cat_columns)
    preds = np.zeros(len(feature_df), dtype=np.float32)
    for path in model_paths:
        model = catboost_mod.CatBoostRegressor()
        model.load_model(str(path))
        fold_preds = model.predict(pool)
        if TRAINING.log_target:
            fold_preds = np.expm1(fold_preds)
        preds += fold_preds.astype(np.float32)
    preds /= max(len(model_paths), 1)
    return preds


def _predict_nn(dense_features: Optional[np.ndarray]) -> np.ndarray:
    if dense_features is None:
        return np.zeros(0, dtype=np.float32)
    model_path = PATHS.models_dir / "nn" / "dense_ensemble.pt"
    if not model_path.exists():
        logging.warning("Dense ensemble model weights not found at %s", model_path)
        return np.zeros(dense_features.shape[0], dtype=np.float32)
    network = DenseEnsembleNetwork(input_dim=dense_features.shape[1])
    torch = import_module("torch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.model.load_state_dict(torch.load(model_path, map_location=device))
    network.model.to(device)
    network.model.eval()
    with torch.no_grad():
        tensor = torch.tensor(dense_features, dtype=torch.float32).to(device)
        preds = network.model(tensor).cpu().numpy().flatten()
    if TRAINING.log_target:
        preds = np.expm1(preds)
    return preds.astype(np.float32)


def _predict_transformer(cache_dir: Path) -> np.ndarray:
    # Placeholder for multimodal transformer predictions.
    predictions_path = PATHS.models_dir / "transformer" / "val_predictions.npy"
    if predictions_path.exists():
        return np.load(predictions_path)
    logging.warning("Transformer predictions not found; returning zeros")
    return np.zeros(pd.read_parquet(PATHS.data_processed / "train_clean.parquet").shape[0], dtype=np.float32)


def _predict_transformer_test(cache_dir: Path) -> np.ndarray:
    predictions_path = PATHS.models_dir / "transformer" / "test_predictions.npy"
    if predictions_path.exists():
        return np.load(predictions_path)
    X_sparse = _prepare_sparse_features(cache_dir)
    return np.zeros(X_sparse.shape[0], dtype=np.float32)


def _compute_ensemble_weights(model_predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, float]:
    available = {name: preds for name, preds in model_predictions.items() if preds is not None and preds.size}
    if not available:
        raise ValueError("No model predictions provided for ensembling")
    names = list(available.keys())
    matrix = np.vstack([available[name] for name in names]).T.astype(np.float64)
    y_true = y_true.astype(np.float64)
    try:
        from scipy.optimize import nnls
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        logging.warning("scipy is required for NNLS weight optimisation: %s", exc)
        weight = 1.0 / len(names)
        return {name: weight for name in names}

    weights, _ = nnls(matrix, y_true)
    if np.isclose(weights.sum(), 0.0):
        weights = np.ones(len(names), dtype=np.float64)
    weights = weights / weights.sum()
    weight_map = {name: float(weight) for name, weight in zip(names, weights)}
    logging.info("Optimised ensemble weights: %s", weight_map)
    return weight_map


@timeit("ensemble")
def build_ensemble(cache_dir: Optional[Path] = None) -> EnsembleResult:
    cache_dir = cache_dir or PATHS.data_processed
    setup_logging()
    ensure_project_structure()

    train_df = pd.read_parquet(PATHS.data_processed / "train_clean.parquet")
    y_true = train_df[TRAINING.target_column].to_numpy(dtype=np.float32)

    gbm_oof = _load_oof_predictions("gbm")
    nn_oof = _load_oof_predictions("nn")
    transformer_oof = _load_oof_predictions("transformer")
    catboost_oof = _load_oof_predictions("catboost")
    oof_predictions = {}
    if gbm_oof is not None and gbm_oof.size:
        oof_predictions["gbm"] = gbm_oof
    if catboost_oof is not None and catboost_oof.size:
        oof_predictions["catboost"] = catboost_oof
    if transformer_oof is not None and transformer_oof.size:
        oof_predictions["transformer"] = transformer_oof
    if nn_oof is not None and nn_oof.size:
        oof_predictions["nn"] = nn_oof

    for name, preds in oof_predictions.items():
        logging.info("%s OOF SMAPE: %.3f", name, smape(y_true, preds))

    weights = _compute_ensemble_weights(oof_predictions, y_true)

    sparse_features_test = _prepare_sparse_features(cache_dir)
    dense_features_test = _prepare_dense_features(cache_dir)
    structured_meta = _load_structured_metadata(cache_dir)
    categorical_columns = structured_meta.get("categorical", [])
    structured_test = _load_structured_features(cache_dir, "test")
    if structured_test is not None:
        structured_columns = structured_meta.get("structured_columns", list(structured_test.columns))
        for column in structured_columns:
            if column not in structured_test.columns:
                if column in categorical_columns:
                    structured_test[column] = "unknown"
                else:
                    structured_test[column] = 0.0
        structured_test = structured_test[structured_columns]

    raw_test_predictions: Dict[str, np.ndarray] = {}
    for name in weights.keys():
        if name == "gbm":
            raw_test_predictions[name] = _predict_gbm(_load_gbm_models(), sparse_features_test)
        elif name == "catboost":
            raw_test_predictions[name] = _predict_catboost(
                _load_catboost_models(),
                structured_test,
                categorical_columns,
            )
        elif name == "transformer":
            raw_test_predictions[name] = _predict_transformer_test(cache_dir)
        elif name == "nn":
            raw_test_predictions[name] = _predict_nn(dense_features_test)

    available_for_test = [name for name, preds in raw_test_predictions.items() if preds is not None and preds.size]
    if not available_for_test:
        raise ValueError("No test predictions generated for ensemble")
    if set(available_for_test) != set(weights.keys()):
        missing = set(weights.keys()) - set(available_for_test)
        logging.warning("Missing test predictions for models: %s", ", ".join(sorted(missing)))
        total = sum(weights[name] for name in available_for_test)
        if np.isclose(total, 0.0):
            adjusted = 1.0 / len(available_for_test)
            for name in available_for_test:
                weights[name] = adjusted
        else:
            for name in available_for_test:
                weights[name] = weights[name] / total
        for name in list(weights.keys()):
            if name not in available_for_test:
                weights.pop(name)
        logging.info("Adjusted ensemble weights after dropping missing models: %s", weights)

    base_length = None
    for preds in raw_test_predictions.values():
        if preds is not None and preds.size:
            base_length = preds.size
            break
    if base_length is None:
        raise ValueError("Unable to determine prediction length for ensemble")

    test_predictions: Dict[str, np.ndarray] = {}
    for name, preds in raw_test_predictions.items():
        if preds is None or preds.size == 0:
            logging.warning("%s test predictions missing; filling with zeros", name)
            test_predictions[name] = np.zeros(base_length, dtype=np.float32)
        elif preds.size != base_length:
            raise ValueError(f"Mismatched prediction length for model {name}: {preds.size} vs {base_length}")
        else:
            test_predictions[name] = preds.astype(np.float32)

    val_blend = np.zeros_like(y_true, dtype=np.float32)
    for name, preds in oof_predictions.items():
        if name in weights:
            val_blend += weights[name] * preds
    val_score = smape(y_true, val_blend)

    test_blend = np.zeros(base_length, dtype=np.float32)
    for name, preds in test_predictions.items():
        test_blend += weights.get(name, 0.0) * preds

    return EnsembleResult(
        validation_predictions=val_blend,
        test_predictions=test_blend.astype(np.float32),
        weights=weights,
        validation_score=val_score,
    )


def save_submission(predictions: np.ndarray, output_path: Optional[Path] = None) -> Path:
    output_path = output_path or PATHS.outputs / "test_out.csv"
    ensure_directory(output_path.parent)
    test_df = pd.read_parquet(PATHS.data_processed / "test_clean.parquet")
    submission = pd.DataFrame({"sample_id": test_df["sample_id"], TRAINING.target_column: predictions})
    submission[TRAINING.target_column] = submission[TRAINING.target_column].clip(lower=1e-3)
    submission.to_csv(output_path, index=False)
    logging.info("Submission saved to %s", output_path)
    return output_path


if __name__ == "__main__":  # pragma: no cover - manual use
    setup_logging()
    result = build_ensemble()
    logging.info("Validation SMAPE %.3f with weights %s", result.validation_score, result.weights)
    save_submission(result.test_predictions)
