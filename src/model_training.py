"""Training routines for multimodal price prediction models."""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import PATHS, TRAINING
from .utils import cache_array, ensure_directory, ensure_project_structure, set_global_seed, setup_logging, smape, timeit


@dataclass
class FeatureMatrices:
    """Container for model-ready feature matrices."""

    X_sparse: Any
    X_dense: Optional[np.ndarray]
    text_embeddings: Optional[np.ndarray]
    image_embeddings: Optional[np.ndarray]
    numeric_features: Optional[np.ndarray]
    structured: Optional[pd.DataFrame]
    categorical_columns: Optional[List[str]]
    numeric_column_names: List[str]
    raw_texts: List[str]
    y: np.ndarray
    sample_ids: np.ndarray
    price_bins: np.ndarray


@dataclass
class GBMConfig:
    """LightGBM hyperparameters."""

    learning_rate: float = 0.05
    max_depth: int = 10
    num_leaves: int = 63
    n_estimators: int = 1000
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_data_in_leaf: int = 20
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1


@dataclass
class CatBoostConfig:
    """CatBoost hyperparameters."""

    iterations: int = 1500
    depth: int = 8
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    random_strength: float = 1.5
    bagging_temperature: float = 1.0
    border_count: int = 128
    od_wait: int = 150
    loss_function: str = "MAE"


@dataclass
class XGBoostConfig:
    """XGBoost hyperparameters."""

    eta: float = 0.05
    max_depth: int = 8
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    n_estimators: int = 1500
    monotone_constraints: Dict[str, int] = field(default_factory=lambda: {"ipq": 1, "avg_listed_price": 1, "mean_numeric_spec": 1})
    early_stopping_rounds: int = 100


@dataclass
class TrainingArtifacts:
    """Paths to trained model assets and validation metrics."""

    gbm_model_paths: List[Path] = field(default_factory=list)
    catboost_model_paths: List[Path] = field(default_factory=list)
    xgboost_model_paths: List[Path] = field(default_factory=list)
    transformer_dir: Optional[Path] = None
    nn_model_path: Optional[Path] = None
    meta_model_path: Optional[Path] = None
    cv_scores: Dict[str, List[float]] = field(default_factory=dict)
    oof_predictions: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class OptunaResult:
    """Container for Optuna hyperparameter tuning output."""

    config: GBMConfig
    best_params: Dict[str, Any]
    best_score: float


def _load_sparse_matrix(path: Path):
    sparse_module = import_module("scipy.sparse")
    return sparse_module.load_npz(path)


def _load_dense_array(path: Path) -> np.ndarray:
    return np.load(path)


def _load_embeddings(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    with np.load(path) as data:
        return data["array"]


def load_feature_matrices(artifact_dir: Path) -> FeatureMatrices:
    from .feature_engineering import FeatureArtifacts

    meta_path = artifact_dir / "structured_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError("Feature metadata missing. Run feature engineering first.")
    with meta_path.open("r", encoding="utf-8") as fp:
        metadata = json.load(fp)
    feature_artifacts = FeatureArtifacts(
        ohe_train_path=artifact_dir / "category_train.npz",
        ohe_test_path=artifact_dir / "category_test.npz",
        numeric_train_path=artifact_dir / "numeric_train.npy",
        numeric_test_path=artifact_dir / "numeric_test.npy",
        text_embedding_train_path=artifact_dir / "text_embeddings_train.npz",
        text_embedding_test_path=artifact_dir / "text_embeddings_test.npz",
        image_embedding_train_path=artifact_dir / "image_embeddings_train.npz",
        image_embedding_test_path=artifact_dir / "image_embeddings_test.npz",
        encoder_path=artifact_dir / "category_encoder.joblib",
        scaler_path=artifact_dir / "numeric_scaler.joblib",
        structured_train_path=artifact_dir / "structured_features_train.parquet",
        structured_test_path=artifact_dir / "structured_features_test.parquet",
    )

    category = _load_sparse_matrix(feature_artifacts.ohe_train_path).tocsr()
    numeric = _load_dense_array(feature_artifacts.numeric_train_path).astype(np.float32)

    sparse_module = import_module("scipy.sparse")
    components = []
    components.append(category)
    components.append(sparse_module.csr_matrix(numeric))

    text_embeddings = _load_embeddings(feature_artifacts.text_embedding_train_path)
    if text_embeddings is not None:
        text_embeddings = text_embeddings.astype(np.float32)
    image_embeddings = _load_embeddings(feature_artifacts.image_embedding_train_path)
    if image_embeddings is not None:
        image_embeddings = image_embeddings.astype(np.float32)
    dense_components: List[np.ndarray] = []
    if text_embeddings is not None:
        dense_components.append(text_embeddings)
        components.append(sparse_module.csr_matrix(text_embeddings))
    if image_embeddings is not None:
        dense_components.append(image_embeddings)
        components.append(sparse_module.csr_matrix(image_embeddings))

    X_sparse = sparse_module.hstack(components).tocsr() if len(components) > 1 else components[0].tocsr()

    X_dense = np.hstack(dense_components) if dense_components else None

    train_df = pd.read_parquet(PATHS.data_processed / "train_clean.parquet")
    y = train_df[TRAINING.target_column].to_numpy(dtype=np.float32)
    sample_ids = train_df["sample_id"].to_numpy()
    price_bins = train_df["price_bin"].to_numpy()
    raw_texts = train_df["catalog_content"].fillna("").astype(str).tolist()
    structured_df = None
    categorical_columns = metadata.get("structured_categorical")
    numeric_column_names = metadata.get("numeric_columns", [])
    structured_path = feature_artifacts.structured_train_path
    structured_columns = metadata.get("structured_columns")
    if structured_path is not None and structured_path.exists():
        structured_df = pd.read_parquet(structured_path)
        if "sample_id" in structured_df.columns:
            structured_df = structured_df.set_index("sample_id")
            structured_df = structured_df.reindex(sample_ids)
        if structured_columns is not None:
            structured_df = structured_df[structured_columns]
        structured_df = structured_df.reset_index(drop=True)

    return FeatureMatrices(
        X_sparse=X_sparse,
        X_dense=X_dense,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        numeric_features=numeric,
        structured=structured_df,
        categorical_columns=categorical_columns,
        numeric_column_names=numeric_column_names,
        raw_texts=raw_texts,
        y=y,
        sample_ids=sample_ids,
        price_bins=price_bins,
    )


class GradientBoostingTrainer:
    """Train LightGBM/CatBoost models on structured features."""

    def __init__(self, config: Optional[GBMConfig] = None, model_dir: Optional[Path] = None, tweedie_variance_power: float = 1.2) -> None:
        self.config = config or GBMConfig()
        self.model_dir = model_dir or PATHS.models_dir / "gbm"
        ensure_directory(self.model_dir)
        try:
            self.lgbm = import_module("lightgbm")
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install lightgbm to train gradient boosting model") from exc
        self.tweedie_variance_power = tweedie_variance_power

    def _build_params(self) -> Dict[str, float]:
        params = {
            "objective": "tweedie",
            "tweedie_variance_power": self.tweedie_variance_power,
            "metric": "mae",
            "learning_rate": self.config.learning_rate,
            "max_depth": self.config.max_depth,
            "num_leaves": self.config.num_leaves,
            "feature_fraction": self.config.colsample_bytree,
            "bagging_fraction": self.config.subsample,
            "bagging_freq": 1,
            "reg_alpha": self.config.reg_alpha,
            "reg_lambda": self.config.reg_lambda,
            "min_data_in_leaf": self.config.min_data_in_leaf,
            "verbose": -1,
            "seed": TRAINING.random_seed,
        }
        return params

    @timeit("train-gbm")
    def train(self, matrices: FeatureMatrices) -> Tuple[List[Path], np.ndarray, List[float]]:
        params = self._build_params()
        sk_module = import_module("sklearn.model_selection")
        StratifiedKFold = getattr(sk_module, "StratifiedKFold")

        kf = StratifiedKFold(
            n_splits=TRAINING.n_splits,
            shuffle=True,
            random_state=TRAINING.random_seed,
        )
        oof_predictions = np.zeros_like(matrices.y, dtype=np.float32)
        model_paths: List[Path] = []
        fold_scores: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(matrices.X_sparse, matrices.price_bins), start=1):
            logging.info("Training GBM fold %s", fold)
            X_train = matrices.X_sparse[train_idx]
            y_train = matrices.y[train_idx]
            X_val = matrices.X_sparse[val_idx]
            y_val = matrices.y[val_idx]

            if TRAINING.log_target:
                y_train_trans = np.log1p(y_train)
                y_val_trans = np.log1p(y_val)
            else:
                y_train_trans = y_train
                y_val_trans = y_val

            train_set = self.lgbm.Dataset(X_train, label=y_train_trans)
            valid_set = self.lgbm.Dataset(X_val, label=y_val_trans, reference=train_set)

            model = self.lgbm.train(
                params,
                train_set,
                valid_sets=[train_set, valid_set],
                num_boost_round=self.config.n_estimators,
                callbacks=[self.lgbm.log_evaluation(100), self.lgbm.early_stopping(stopping_rounds=200)],
            )

            preds_trans = model.predict(X_val, num_iteration=model.best_iteration)
            if TRAINING.log_target:
                preds = np.expm1(preds_trans)
                y_val_true = y_val
            else:
                preds = preds_trans
                y_val_true = y_val
            oof_predictions[val_idx] = preds.astype(np.float32)
            fold_score = smape(y_val_true, preds)
            fold_scores.append(fold_score)
            logging.info("Fold %s SMAPE: %.3f", fold, fold_score)

            model_path = self.model_dir / f"lightgbm_fold{fold}.txt"
            model.save_model(str(model_path))
            model_paths.append(model_path)

        logging.info("GBM CV SMAPE: %.3f", float(np.mean(fold_scores)))
        return model_paths, oof_predictions, fold_scores


class CatBoostTrainer:
    """Train CatBoost models on structured tabular features."""

    def __init__(self, config: Optional[CatBoostConfig] = None, model_dir: Optional[Path] = None) -> None:
        self.config = config or CatBoostConfig()
        self.model_dir = model_dir or PATHS.models_dir / "catboost"
        ensure_directory(self.model_dir)
        try:
            self.catboost = import_module("catboost")
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install catboost to train CatBoost models") from exc

    def _build_params(self) -> Dict[str, Any]:
        return {
            "iterations": self.config.iterations,
            "depth": self.config.depth,
            "learning_rate": self.config.learning_rate,
            "l2_leaf_reg": self.config.l2_leaf_reg,
            "random_strength": self.config.random_strength,
            "bagging_temperature": self.config.bagging_temperature,
            "border_count": self.config.border_count,
            "loss_function": self.config.loss_function,
            "eval_metric": "MAE",
            "od_type": "Iter",
            "od_wait": self.config.od_wait,
            "verbose": False,
            "random_seed": TRAINING.random_seed,
            "allow_writing_files": False,
        }

    @timeit("train-catboost")
    def train(self, matrices: FeatureMatrices) -> Tuple[List[Path], np.ndarray, List[float]]:
        params = self._build_params()

        numeric_data = matrices.numeric_features.astype(np.float32) if matrices.numeric_features is not None else None
        cat_data = None
        cat_columns = [col for col in (matrices.categorical_columns or [])]
        if matrices.structured is not None and cat_columns:
            cat_df = matrices.structured[cat_columns].fillna("unknown").astype(str)
            cat_data = cat_df.to_numpy()
        text_data = matrices.raw_texts
        embedding_data = []
        if matrices.text_embeddings is not None:
            embedding_data.append(matrices.text_embeddings)
        if matrices.image_embeddings is not None:
            embedding_data.append(matrices.image_embeddings)

        if numeric_data is None and not embedding_data and cat_data is None and not text_data:
            raise ValueError("No features available for CatBoost training")

        sk_module = import_module("sklearn.model_selection")
        StratifiedKFold = getattr(sk_module, "StratifiedKFold")
        kf = StratifiedKFold(
            n_splits=TRAINING.n_splits,
            shuffle=True,
            random_state=TRAINING.random_seed,
        )

        Pool = getattr(self.catboost, "Pool")
        FeaturesData = getattr(self.catboost, "FeaturesData")

        def make_pool(indices: np.ndarray):
            num_subset = numeric_data[indices] if numeric_data is not None else None
            cat_subset = cat_data[indices] if cat_data is not None else None
            text_subset = [text_data[i] for i in indices]
            embedding_subset = [emb[indices] for emb in embedding_data] if embedding_data else None
            features = FeaturesData(
                num_feature_data=num_subset,
                cat_feature_data=cat_subset,
                text_feature_data=text_subset,
                embedding_feature_data=embedding_subset,
            )
            label = matrices.y[indices]
            if TRAINING.log_target:
                label = np.log1p(label)
            return Pool(data=features, label=label)

        oof_predictions = np.zeros_like(matrices.y, dtype=np.float32)
        fold_scores: List[float] = []
        model_paths: List[Path] = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(matrices.y, matrices.price_bins), start=1):
            logging.info("Training CatBoost fold %s", fold)
            train_pool = make_pool(train_idx)
            val_pool = make_pool(val_idx)

            model = self.catboost.CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=val_pool, verbose=False, use_best_model=True)

            preds = np.asarray(model.predict(val_pool), dtype=np.float64)
            if TRAINING.log_target:
                preds = np.expm1(preds)
            preds = preds.astype(np.float32)
            y_val_true = matrices.y[val_idx]
            fold_score = smape(y_val_true, preds)
            fold_scores.append(fold_score)
            logging.info("CatBoost Fold %s SMAPE: %.3f", fold, fold_score)

            oof_predictions[val_idx] = preds

            model_path = self.model_dir / f"catboost_fold{fold}.cbm"
            model.save_model(str(model_path))
            model_paths.append(model_path)

        logging.info("CatBoost CV SMAPE: %.3f", float(np.mean(fold_scores)))
        return model_paths, oof_predictions, fold_scores


class XGBoostTrainer:
    """Train XGBoost gradient boosted trees with optional monotonic constraints."""

    def __init__(self, config: Optional[XGBoostConfig] = None, model_dir: Optional[Path] = None) -> None:
        self.config = config or XGBoostConfig()
        self.model_dir = model_dir or PATHS.models_dir / "xgboost"
        ensure_directory(self.model_dir)
        try:
            self.xgb = import_module("xgboost")
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install xgboost to train XGBoost models") from exc

    def _prepare_features(self, matrices: FeatureMatrices) -> Tuple[np.ndarray, List[str]]:
        numeric = matrices.numeric_features
        if numeric is None:
            raise ValueError("Numeric features are required for XGBoost training")

        feature_names = list(matrices.numeric_column_names or [])
        if len(feature_names) != numeric.shape[1] or len(set(feature_names)) != len(feature_names):
            feature_names = [f"f_{idx}" for idx in range(numeric.shape[1])]

        numeric = np.nan_to_num(numeric.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return numeric, feature_names

    def _build_params(self, feature_names: List[str]) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "eta": self.config.eta,
            "max_depth": self.config.max_depth,
            "subsample": self.config.subsample,
            "colsample_bytree": self.config.colsample_bytree,
            "alpha": self.config.reg_alpha,
            "lambda": self.config.reg_lambda,
            "tree_method": "hist",
            "verbosity": 0,
            "seed": TRAINING.random_seed,
        }

        monotone_vector = [int(self.config.monotone_constraints.get(name, 0)) for name in feature_names]
        if any(monotone_vector):
            params["monotone_constraints"] = "(" + ",".join(str(value) for value in monotone_vector) + ")"
        return params

    @timeit("train-xgboost")
    def train(self, matrices: FeatureMatrices) -> Tuple[List[Path], np.ndarray, List[float]]:
        X_numeric, feature_names = self._prepare_features(matrices)
        params = self._build_params(feature_names)

        sk_module = import_module("sklearn.model_selection")
        StratifiedKFold = getattr(sk_module, "StratifiedKFold")
        kf = StratifiedKFold(
            n_splits=TRAINING.n_splits,
            shuffle=True,
            random_state=TRAINING.random_seed,
        )

        oof_predictions = np.zeros_like(matrices.y, dtype=np.float32)
        fold_scores: List[float] = []
        model_paths: List[Path] = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_numeric, matrices.price_bins), start=1):
            logging.info("Training XGBoost fold %s", fold)
            X_train = X_numeric[train_idx]
            X_val = X_numeric[val_idx]
            y_train = matrices.y[train_idx]
            y_val = matrices.y[val_idx]

            if TRAINING.log_target:
                y_train_fit = np.log1p(y_train)
                y_val_fit = np.log1p(y_val)
            else:
                y_train_fit = y_train
                y_val_fit = y_val

            dtrain = self.xgb.DMatrix(X_train, label=y_train_fit, feature_names=feature_names)
            dval = self.xgb.DMatrix(X_val, label=y_val_fit, feature_names=feature_names)

            booster = self.xgb.train(
                params,
                dtrain,
                num_boost_round=self.config.n_estimators,
                evals=[(dtrain, "train"), (dval, "valid")],
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose_eval=False,
            )

            best_ntree_limit = getattr(booster, "best_ntree_limit", 0)
            if best_ntree_limit:
                val_preds_trans = booster.predict(dval, ntree_limit=best_ntree_limit)
            else:
                val_preds_trans = booster.predict(dval)

            if TRAINING.log_target:
                val_preds = np.expm1(val_preds_trans)
            else:
                val_preds = val_preds_trans

            oof_predictions[val_idx] = val_preds.astype(np.float32)
            fold_score = smape(y_val, val_preds)
            fold_scores.append(fold_score)
            logging.info("XGBoost Fold %s SMAPE: %.3f", fold, fold_score)

            model_path = self.model_dir / f"xgboost_fold{fold}.json"
            booster.save_model(str(model_path))
            model_paths.append(model_path)

        logging.info("XGBoost CV SMAPE: %.3f", float(np.mean(fold_scores)))
        return model_paths, oof_predictions, fold_scores


def tune_gbm_hyperparameters(
    matrices: FeatureMatrices,
    n_trials: int = 30,
    timeout: Optional[int] = None,
    n_splits: Optional[int] = None,
) -> OptunaResult:
    """Use Optuna to optimise LightGBM hyperparameters for SMAPE."""

    try:
        import optuna
        from optuna.exceptions import TrialPruned
    except ModuleNotFoundError as exc:
        raise RuntimeError("Optuna is required for hyperparameter tuning. Install it via pip install optuna.") from exc

    lgbm = import_module("lightgbm")
    sk_module = import_module("sklearn.model_selection")
    StratifiedKFold = getattr(sk_module, "StratifiedKFold")

    n_splits = n_splits or TRAINING.n_splits
    logging.info("Starting Optuna tuning: %s trials, timeout=%s", n_trials, timeout)

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 6, 16),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 150),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
        }
        num_boost_round = trial.suggest_int("n_estimators", 600, 2000)

        base_params = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "bagging_freq": 1,
            "seed": TRAINING.random_seed,
        }
        params_full = {**base_params, **params}

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=TRAINING.random_seed)
        fold_scores: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(matrices.X_sparse, matrices.price_bins), start=1):
            X_train = matrices.X_sparse[train_idx]
            y_train = matrices.y[train_idx]
            X_val = matrices.X_sparse[val_idx]
            y_val = matrices.y[val_idx]

            if TRAINING.log_target:
                y_train = np.log1p(y_train)
                y_val = np.log1p(y_val)

            train_set = lgbm.Dataset(X_train, label=y_train)
            valid_set = lgbm.Dataset(X_val, label=y_val, reference=train_set)

            model = lgbm.train(
                params_full,
                train_set,
                valid_sets=[valid_set],
                num_boost_round=num_boost_round,
                callbacks=[
                    lgbm.early_stopping(stopping_rounds=150, verbose=False),
                ],
            )

            preds = model.predict(X_val, num_iteration=model.best_iteration)
            if TRAINING.log_target:
                preds = np.expm1(preds)
                y_val_true = np.expm1(y_val)
            else:
                y_val_true = y_val

            fold_score = smape(y_val_true, preds)
            fold_scores.append(fold_score)
            trial.report(fold_score, step=fold_idx)
            if trial.should_prune():
                raise TrialPruned()

        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="minimize", study_name="lightgbm_smape")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)

    logging.info(
        "Optuna completed: best SMAPE %.4f over %s trials (%s pruned)",
        study.best_value,
        len(study.trials),
        len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    )
    best_params = study.best_params
    best_config = GBMConfig(
        learning_rate=float(best_params["learning_rate"]),
        max_depth=int(best_params["max_depth"]),
        num_leaves=int(best_params["num_leaves"]),
        n_estimators=int(best_params["n_estimators"]),
        subsample=float(best_params["bagging_fraction"]),
        colsample_bytree=float(best_params["feature_fraction"]),
        min_data_in_leaf=int(best_params["min_data_in_leaf"]),
        reg_alpha=float(best_params["lambda_l1"]),
        reg_lambda=float(best_params["lambda_l2"]),
    )

    ensure_directory(PATHS.models_dir)
    best_path = PATHS.models_dir / "gbm_optuna_best.json"
    with best_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "best_value": study.best_value,
                "best_params": best_params,
                "n_trials": len(study.trials),
            },
            fp,
            indent=2,
        )
    logging.info("Saved best Optuna params to %s", best_path)

    return OptunaResult(config=best_config, best_params=best_params, best_score=float(study.best_value))


class DenseEnsembleNetwork:
    """Simple feed-forward network over dense embeddings."""

    def __init__(self, input_dim: int, model_dir: Optional[Path] = None) -> None:
        try:
            torch = import_module("torch")
            nn = import_module("torch.nn")
            optim = import_module("torch.optim")
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("Install torch to train the dense ensemble network") from exc
        self.torch = torch
        self.nn = nn
        self.optim = optim
        self.model_dir = model_dir or PATHS.models_dir / "nn"
        ensure_directory(self.model_dir)
        self.model = self._build_model(input_dim)

    def _build_model(self, input_dim: int):
        layers = [
            self.nn.Linear(input_dim, 512),
            self.nn.BatchNorm1d(512),
            self.nn.ReLU(),
            self.nn.Dropout(0.3),
            self.nn.Linear(512, 256),
            self.nn.BatchNorm1d(256),
            self.nn.ReLU(),
            self.nn.Dropout(0.3),
            self.nn.Linear(256, 128),
            self.nn.BatchNorm1d(128),
            self.nn.ReLU(),
            self.nn.Dropout(0.3),
            self.nn.Linear(128, 1),
        ]
        model = self.nn.Sequential(*layers)
        return model

    @timeit("train-nn")
    def train(self, X: np.ndarray, y: np.ndarray, price_bins: np.ndarray) -> Tuple[Path, np.ndarray, List[float]]:
        device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        self.model.to(device)
        criterion = self.nn.SmoothL1Loss()
        optimizer = self.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)

        sk_module = import_module("sklearn.model_selection")
        StratifiedKFold = getattr(sk_module, "StratifiedKFold")
        kf = StratifiedKFold(n_splits=TRAINING.n_splits, shuffle=True, random_state=TRAINING.random_seed)

        oof_preds = np.zeros_like(y, dtype=np.float32)
        fold_scores: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, price_bins), start=1):
            logging.info("Training NN fold %s", fold)
            X_train = self.torch.tensor(X[train_idx], dtype=self.torch.float32).to(device)
            y_train = self.torch.tensor(np.log1p(y[train_idx]) if TRAINING.log_target else y[train_idx], dtype=self.torch.float32).unsqueeze(1).to(device)
            X_val = self.torch.tensor(X[val_idx], dtype=self.torch.float32).to(device)
            y_val = self.torch.tensor(np.log1p(y[val_idx]) if TRAINING.log_target else y[val_idx], dtype=self.torch.float32).unsqueeze(1).to(device)

            best_model_state = None
            best_score = float("inf")
            patience = 5
            wait = 0

            for epoch in range(1, 51):
                self.model.train()
                optimizer.zero_grad()
                preds = self.model(X_train)
                loss = criterion(preds, y_train)
                loss.backward()
                optimizer.step()

                self.model.eval()
                with self.torch.no_grad():
                    val_preds = self.model(X_val)
                    val_loss = criterion(val_preds, y_val).item()
                logging.info("Fold %s Epoch %s Loss %.4f ValLoss %.4f", fold, epoch, loss.item(), val_loss)

                if val_loss + 1e-4 < best_score:
                    best_score = val_loss
                    best_model_state = self.model.state_dict()
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        logging.info("Early stopping on fold %s at epoch %s", fold, epoch)
                        break

            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

            self.model.eval()
            with self.torch.no_grad():
                val_preds = self.model(X_val).cpu().numpy().flatten()
            if TRAINING.log_target:
                val_preds = np.expm1(val_preds)
                y_val_true = np.expm1(y_val.cpu().numpy().flatten())
            else:
                y_val_true = y_val.cpu().numpy().flatten()
            oof_preds[val_idx] = val_preds.astype(np.float32)
            fold_score = smape(y_val_true, val_preds)
            fold_scores.append(fold_score)
            logging.info("NN Fold %s SMAPE: %.3f", fold, fold_score)

        model_path = self.model_dir / "dense_ensemble.pt"
        self.torch.save(self.model.state_dict(), model_path)
        return model_path, oof_preds, fold_scores


def train_models(
    feature_dir: Optional[Path] = None,
    matrices: Optional[FeatureMatrices] = None,
    gbm_config: Optional[GBMConfig] = None,
    catboost_config: Optional[CatBoostConfig] = None,
    train_catboost: bool = True,
    xgboost_config: Optional[XGBoostConfig] = None,
    train_xgboost: bool = True,
) -> TrainingArtifacts:
    setup_logging()
    set_global_seed(TRAINING.random_seed)
    ensure_project_structure()

    artifact_dir = feature_dir or PATHS.data_processed
    matrices = matrices or load_feature_matrices(artifact_dir)

    artifacts = TrainingArtifacts()

    if gbm_config is not None:
        logging.info("Training GBM with tuned hyperparameters: %s", gbm_config)
    gbm_trainer = GradientBoostingTrainer(config=gbm_config)
    gbm_paths, gbm_oof, gbm_scores = gbm_trainer.train(matrices)
    artifacts.gbm_model_paths = gbm_paths
    artifacts.cv_scores["gbm"] = gbm_scores
    artifacts.oof_predictions["gbm"] = gbm_oof

    catboost_config_used: Optional[CatBoostConfig] = None
    if train_catboost:
        if matrices.structured is None or matrices.structured.empty:
            logging.warning("Structured features missing; skipping CatBoost model")
        else:
            try:
                catboost_trainer = CatBoostTrainer(config=catboost_config)
            except (RuntimeError, ValueError) as exc:
                logging.warning("Skipping CatBoost training: %s", exc)
            else:
                catboost_config_used = catboost_trainer.config
                catboost_paths, catboost_oof, catboost_scores = catboost_trainer.train(matrices)
                artifacts.catboost_model_paths = catboost_paths
                artifacts.cv_scores["catboost"] = catboost_scores
                artifacts.oof_predictions["catboost"] = catboost_oof
    else:
        logging.info("CatBoost training disabled via --skip-catboost flag")

    xgboost_config_used: Optional[XGBoostConfig] = None
    if train_xgboost:
        try:
            xgboost_trainer = XGBoostTrainer(config=xgboost_config)
        except (RuntimeError, ValueError) as exc:
            logging.warning("Skipping XGBoost training: %s", exc)
        else:
            xgboost_config_used = xgboost_trainer.config
            xgb_paths, xgb_oof, xgb_scores = xgboost_trainer.train(matrices)
            artifacts.xgboost_model_paths = xgb_paths
            artifacts.cv_scores["xgboost"] = xgb_scores
            artifacts.oof_predictions["xgboost"] = xgb_oof
    else:
        logging.info("XGBoost training disabled via flag")

    if matrices.X_dense is not None:
        nn_trainer = DenseEnsembleNetwork(input_dim=matrices.X_dense.shape[1])
        nn_path, nn_oof, nn_scores = nn_trainer.train(matrices.X_dense, matrices.y, matrices.price_bins)
        artifacts.nn_model_path = nn_path
        artifacts.cv_scores["nn"] = nn_scores
        artifacts.oof_predictions["nn"] = nn_oof
    else:
        logging.warning("Dense embeddings not available; skipping neural network model")

    metrics_path = PATHS.models_dir / "cv_metrics.json"
    ensure_directory(metrics_path.parent)
    summary = {
        model: {
            "scores": [float(score) for score in scores],
            "mean": float(np.mean(scores)),
        }
        for model, scores in artifacts.cv_scores.items()
    }
    if gbm_config is not None and "gbm" in summary:
        summary["gbm"]["params"] = asdict(gbm_config)
    if catboost_config_used is not None and "catboost" in summary:
        summary["catboost"]["params"] = asdict(catboost_config_used)
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    logging.info("Validation metrics stored at %s", metrics_path)

    base_model_names = [
        name
        for name in ("gbm", "catboost", "xgboost", "transformer", "nn")
        if name in artifacts.oof_predictions and artifacts.oof_predictions[name].size
    ]

    if base_model_names:
        X_meta = np.column_stack([artifacts.oof_predictions[name] for name in base_model_names])
        X_meta = np.clip(X_meta, a_min=0.0, a_max=None)
        X_meta_trans = np.log1p(X_meta)
        y_meta = matrices.y
        if TRAINING.log_target:
            y_meta_trans = np.log1p(y_meta)
        else:
            y_meta_trans = y_meta

        meta_pipeline = make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=1e-4))
        meta_pipeline.fit(X_meta_trans, y_meta_trans)
        meta_preds_trans = meta_pipeline.predict(X_meta_trans)
        if TRAINING.log_target:
            meta_preds = np.expm1(meta_preds_trans)
        else:
            meta_preds = meta_preds_trans
        meta_preds = np.clip(meta_preds, a_min=1e-3, a_max=None).astype(np.float32)
        meta_score = smape(matrices.y, meta_preds)
        artifacts.oof_predictions["meta"] = meta_preds
        artifacts.cv_scores.setdefault("meta", []).append(meta_score)
        logging.info("Meta-model SMAPE: %.3f", meta_score)

        meta_dir = PATHS.models_dir / "meta"
        ensure_directory(meta_dir)
        meta_model_path = meta_dir / "meta_regressor.joblib"
        joblib.dump(meta_pipeline, meta_model_path)
        artifacts.meta_model_path = meta_model_path
        meta_config = {
            "base_models": base_model_names,
            "log_features": True,
            "log_target": TRAINING.log_target,
            "coefficients": meta_pipeline.named_steps["huberregressor"].coef_.tolist(),
        }
        with (meta_dir / "meta_config.json").open("w", encoding="utf-8") as fp:
            json.dump(meta_config, fp, indent=2)
        cache_array(meta_preds, PATHS.models_dir / "oof_meta.npz")

    for model_name, preds in artifacts.oof_predictions.items():
        cache_array(preds, PATHS.models_dir / f"oof_{model_name}.npz")

    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multimodal price prediction models")
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=None,
        help="Directory containing cached features (defaults to data/processed)",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="Number of Optuna trials to run for LightGBM hyperparameter tuning",
    )
    parser.add_argument(
        "--optuna-timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds for the Optuna study",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training and only perform hyperparameter search",
    )
    parser.add_argument(
        "--skip-catboost",
        action="store_true",
        help="Skip CatBoost model training",
    )
    args = parser.parse_args()

    feature_dir = args.feature_dir or PATHS.data_processed
    matrices: Optional[FeatureMatrices] = None
    gbm_config: Optional[GBMConfig] = None

    if args.optuna_trials > 0:
        setup_logging()
        ensure_project_structure()
        set_global_seed(TRAINING.random_seed)
        matrices = load_feature_matrices(feature_dir)
        optuna_result = tune_gbm_hyperparameters(
            matrices,
            n_trials=args.optuna_trials,
            timeout=args.optuna_timeout,
        )
        gbm_config = optuna_result.config
        logging.info(
            "Optuna best SMAPE %.4f with params %s",
            optuna_result.best_score,
            optuna_result.best_params,
        )

    if not args.skip_train:
        train_models(
            feature_dir=feature_dir,
            matrices=matrices,
            gbm_config=gbm_config,
            train_catboost=not args.skip_catboost,
        )
