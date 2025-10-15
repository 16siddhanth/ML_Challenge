"""Feature extraction logic for text and images."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import PATHS
from .utils import cache_array, download_image_with_retry, ensure_directory, ensure_project_structure, load_image, setup_logging, timeit

_BRAND_PATTERN = re.compile(r"(?:brand\s*[:\-]?\s*)(?P<brand>[A-Za-z0-9&'\s]+)", re.IGNORECASE)
_BRAND_TOKEN_PATTERN = re.compile(r"\b([A-Z][a-zA-Z0-9&']{2,})\b")
_IPQ_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:pack|pc|pcs|count|ct|pk|bundle|set)", re.IGNORECASE)
_NUMERIC_PATTERN = re.compile(r"(\d+(?:\.\d+)?)(?=\s*(?:ml|g|kg|oz|lb|inch|in|cm|mm|l|litre|liter|ft|meter|mph|hz|mah|w|kw|mp|dpi))", re.IGNORECASE)
_PRICE_PATTERN = re.compile(r"([$€£₹]|usd|inr|eur|gbp)\s*(\d+(?:[.,]\d+)?)(?!\w)", re.IGNORECASE)
_CATEGORY_SEPARATORS = re.compile(r"[\|>/\\]")

_DEFAULT_IMAGE_SIZE = (224, 224)


@dataclass
class FeatureArtifacts:
    """Paths to stored feature matrices and fitted transformers."""

    ohe_train_path: Path
    ohe_test_path: Path
    numeric_train_path: Path
    numeric_test_path: Path
    text_embedding_train_path: Optional[Path]
    text_embedding_test_path: Optional[Path]
    image_embedding_train_path: Optional[Path]
    image_embedding_test_path: Optional[Path]
    encoder_path: Path
    scaler_path: Path
    structured_train_path: Optional[Path]
    structured_test_path: Optional[Path]


def _normalize_brand(raw: str) -> str:
    brand = raw.lower().strip()
    brand = re.sub(r"[^a-z0-9& ]", "", brand)
    return brand or "unknown"


def extract_brand(text: str) -> str:
    match = _BRAND_PATTERN.search(text)
    if match:
        return _normalize_brand(match.group("brand"))
    tokens = _BRAND_TOKEN_PATTERN.findall(text)
    if tokens:
        return _normalize_brand(tokens[0])
    return "unknown"


def parse_ipq(text: str) -> float:
    match = _IPQ_PATTERN.search(text)
    if not match:
        return 1.0
    try:
        return float(match.group(1))
    except ValueError:
        return 1.0


def extract_numeric_specs(text: str) -> float:
    values = []
    for value in _NUMERIC_PATTERN.findall(text):
        try:
            values.append(float(value))
        except ValueError:
            continue
    if not values:
        return 0.0
    return float(np.mean(values))


def extract_price_tokens(text: str) -> Dict[str, float]:
    matches = _PRICE_PATTERN.findall(text)
    if not matches:
        return {"has_currency": 0, "avg_listed_price": 0.0}
    values = []
    for _, value in matches:
        value = value.replace(",", "")
        try:
            values.append(float(value))
        except ValueError:
            continue
    if not values:
        return {"has_currency": 1, "avg_listed_price": 0.0}
    return {"has_currency": 1, "avg_listed_price": float(np.mean(values))}


def extract_keywords(text: str) -> Dict[str, int]:
    lowered = text.lower()
    return {
        "is_premium": int(any(token in lowered for token in ("luxury", "premium", "exclusive", "designer"))),
        "is_budget": int(any(token in lowered for token in ("budget", "value", "discount", "cheap"))),
        "is_bundle": int(any(token in lowered for token in ("bundle", "pack", "set", "combo"))),
        "is_limited": int(any(token in lowered for token in ("limited", "special edition", "collector"))),
        "is_refurb": int(any(token in lowered for token in ("refurb", "renewed", "used"))),
    }


def clean_category(raw: str) -> str:
    tokens = _CATEGORY_SEPARATORS.split(raw)
    tokens = [tok.strip().lower() for tok in tokens if tok.strip()]
    if not tokens:
        return "unknown"
    return tokens[-1]


def build_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Deriving structured text features from catalog content")
    catalog = df["catalog_content"].fillna("").astype(str)
    base = pd.DataFrame(
        {
            "brand": catalog.apply(extract_brand),
            "ipq": catalog.apply(parse_ipq),
            "mean_numeric_spec": catalog.apply(extract_numeric_specs),
            "char_len": catalog.str.len(),
            "word_len": catalog.str.split().apply(len),
            "unique_word_ratio": catalog.apply(lambda text: len(set(text.split())) / max(len(text.split()), 1)),
            "num_sentences": catalog.str.count(r"[.!?]") + 1,
        }
    )
    keyword_df = catalog.apply(extract_keywords).apply(pd.Series)
    base[keyword_df.columns] = keyword_df
    price_df = catalog.apply(extract_price_tokens).apply(pd.Series)
    base[price_df.columns] = price_df
    if "product_type" in df.columns:
        base["category"] = df["product_type"].astype(str).apply(clean_category)
    else:
        base["category"] = catalog.apply(lambda text: clean_category(text[:80]))
    return base
def _get_one_hot_encoder() -> object:
    try:
        prep_module = import_module("sklearn.preprocessing")
        OneHotEncoder = getattr(prep_module, "OneHotEncoder")
    except (ModuleNotFoundError, AttributeError) as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for categorical encoding") from exc
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # legacy scikit-learn versions
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _get_robust_scaler() -> object:
    try:
        prep_module = import_module("sklearn.preprocessing")
        RobustScaler = getattr(prep_module, "RobustScaler")
    except (ModuleNotFoundError, AttributeError) as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for numeric scaling") from exc
    return RobustScaler()


class TextEmbeddingModel:
    """Wrapper for transformer-based sentence embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens") -> None:
        try:
            sentence_module = import_module("sentence_transformers")
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install sentence-transformers to compute text embeddings") from exc
        SentenceTransformer = getattr(sentence_module, "SentenceTransformer")
        try:
            torch = import_module("torch")
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ModuleNotFoundError:  # pragma: no cover - optional dependency
            device = "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.device = device

    def encode(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        items = list(texts)
        embeddings = self.model.encode(
            items,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)


class ImageEmbeddingModel:
    """Wrapper for CLIP-style vision encoders."""

    def __init__(self, model_name: str = "ViT-B-32", image_size: Tuple[int, int] = _DEFAULT_IMAGE_SIZE) -> None:
        try:
            self.open_clip = import_module("open_clip")
            self.torch = import_module("torch")
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install open-clip-pytorch and torch for image embeddings") from exc
        self.model_name = model_name
        self.image_size = image_size
        self.model, _, self.preprocess = self.open_clip.create_model_and_transforms(model_name)
        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.output_dim = int(getattr(self.model.visual, "output_dim"))

    def encode(self, image_paths: Iterable[Path], batch_size: int = 32) -> np.ndarray:
        embeddings: list[np.ndarray] = []
        buffer: list = []
        expected = 0
        with self.torch.no_grad():
            for path in tqdm(list(image_paths), desc="Image embeddings"):
                expected += 1
                image = load_image(path, size=self.image_size)
                if image is None:
                    embeddings.append(np.zeros(self.output_dim, dtype=np.float32))
                    continue
                tensor = self.preprocess(image).unsqueeze(0)
                buffer.append(tensor)
                if len(buffer) == batch_size:
                    batch_tensor = self.torch.cat(buffer).to(self.device)
                    feats = self.model.encode_image(batch_tensor)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    embeddings.extend(feats.cpu().numpy().astype(np.float32))
                    buffer = []
            if buffer:
                batch_tensor = self.torch.cat(buffer).to(self.device)
                feats = self.model.encode_image(batch_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                embeddings.extend(feats.cpu().numpy().astype(np.float32))
        if len(embeddings) < expected:
            padding = expected - len(embeddings)
            embeddings.extend([np.zeros(self.output_dim, dtype=np.float32) for _ in range(padding)])
        return np.vstack(embeddings)


def _save_sparse_matrix(path: Path, matrix) -> None:
    try:
        sparse_module = import_module("scipy.sparse")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required to persist sparse matrices") from exc
    sparse = sparse_module
    ensure_directory(path.parent)
    sparse.save_npz(path, matrix)


@timeit("feature-engineering")
def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cache_dir: Path = PATHS.data_processed,
    compute_text_embeddings: bool = True,
    compute_image_embeddings: bool = False,
    image_cache_dir: Optional[Path] = None,
) -> FeatureArtifacts:
    ensure_directory(cache_dir)
    n_train = len(train_df)

    train_struct = build_structured_features(train_df)
    test_struct = build_structured_features(test_df)

    structured_train_export = train_struct.copy()
    structured_train_export["sample_id"] = train_df["sample_id"].to_numpy()
    structured_test_export = test_struct.copy()
    structured_test_export["sample_id"] = test_df["sample_id"].to_numpy()

    train_texts = train_df["catalog_content"].fillna("").astype(str)
    test_texts = test_df["catalog_content"].fillna("").astype(str)

    encoder = _get_one_hot_encoder()
    ohe_train = encoder.fit_transform(train_struct[["category"]])
    ohe_test = encoder.transform(test_struct[["category"]])

    scaler = _get_robust_scaler()
    numeric_columns = [
        "ipq",
        "mean_numeric_spec",
        "char_len",
        "word_len",
        "unique_word_ratio",
        "num_sentences",
        "is_premium",
        "is_budget",
        "is_bundle",
        "is_limited",
        "is_refurb",
        "has_currency",
        "avg_listed_price",
    ]
    numeric_train = scaler.fit_transform(train_struct[numeric_columns])
    numeric_test = scaler.transform(test_struct[numeric_columns])

    ohe_train_path = cache_dir / "category_train.npz"
    ohe_test_path = cache_dir / "category_test.npz"
    _save_sparse_matrix(ohe_train_path, ohe_train)
    _save_sparse_matrix(ohe_test_path, ohe_test)

    numeric_train_path = cache_dir / "numeric_train.npy"
    numeric_test_path = cache_dir / "numeric_test.npy"
    np.save(numeric_train_path, numeric_train.astype(np.float32), allow_pickle=False)
    np.save(numeric_test_path, numeric_test.astype(np.float32), allow_pickle=False)

    text_embedding_train_path = None
    text_embedding_test_path = None
    text_model_name = None
    text_embedding_dim: Optional[int] = None
    if compute_text_embeddings:
        text_model = TextEmbeddingModel()
        text_model_name = text_model.model_name
        embeddings_train = text_model.encode(train_texts)
        embeddings_test = text_model.encode(test_texts)
        text_embedding_dim = int(embeddings_train.shape[1]) if embeddings_train.ndim == 2 else None
        text_embedding_train_path = cache_dir / "text_embeddings_train.npz"
        text_embedding_test_path = cache_dir / "text_embeddings_test.npz"
        cache_array(embeddings_train, text_embedding_train_path)
        cache_array(embeddings_test, text_embedding_test_path)
    else:
        logging.warning("Text embeddings disabled; DistilBERT features will be unavailable")

    image_embedding_train_path = None
    image_embedding_test_path = None
    image_model_name = None
    if compute_image_embeddings:
        if image_cache_dir is None:
            image_cache_dir = PATHS.data_processed / "images"
        ensure_directory(image_cache_dir)
        combined = pd.concat([train_df, test_df], ignore_index=True)
        image_paths: list[Path] = []
        for row in combined.itertuples():
            image_path = image_cache_dir / f"{row.sample_id}.jpg"
            if not image_path.exists() and getattr(row, "image_link", ""):
                download_image_with_retry(row.image_link, image_path)
            image_paths.append(image_path)
        image_model = ImageEmbeddingModel()
        image_model_name = image_model.model_name
        embeddings = image_model.encode(image_paths)
        image_embedding_train_path = cache_dir / "image_embeddings_train.npz"
        image_embedding_test_path = cache_dir / "image_embeddings_test.npz"
        cache_array(embeddings[:n_train], image_embedding_train_path)
        cache_array(embeddings[n_train:], image_embedding_test_path)

    structured_train_path = cache_dir / "structured_features_train.parquet"
    structured_test_path = cache_dir / "structured_features_test.parquet"
    structured_train_export.to_parquet(structured_train_path, index=False)
    structured_test_export.to_parquet(structured_test_path, index=False)

    encoder_path = cache_dir / "category_encoder.joblib"
    scaler_path = cache_dir / "numeric_scaler.joblib"
    joblib.dump(encoder, encoder_path)
    joblib.dump(scaler, scaler_path)

    metadata_path = cache_dir / "structured_metadata.json"
    metadata = {
        "numeric_columns": numeric_columns,
        "text_model": text_model_name,
        "image_model": image_model_name,
        "structured_columns": [col for col in structured_train_export.columns if col != "sample_id"],
        "structured_categorical": [
            col
            for col, dtype in structured_train_export.drop(columns=["sample_id"]).dtypes.items()
            if dtype == object or str(dtype).startswith("category")
        ],
        "text_embedding_dim": text_embedding_dim,
    }
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    return FeatureArtifacts(
        ohe_train_path=ohe_train_path,
        ohe_test_path=ohe_test_path,
        numeric_train_path=numeric_train_path,
        numeric_test_path=numeric_test_path,
        text_embedding_train_path=text_embedding_train_path,
        text_embedding_test_path=text_embedding_test_path,
        image_embedding_train_path=image_embedding_train_path,
        image_embedding_test_path=image_embedding_test_path,
        encoder_path=encoder_path,
        scaler_path=scaler_path,
        structured_train_path=structured_train_path,
        structured_test_path=structured_test_path,
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation
    import argparse

    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument(
        "--skip-text-embeddings",
        action="store_true",
        help="Disable DistilBERT text embeddings (enabled by default)",
    )
    parser.add_argument("--image-embeddings", action="store_true", help="Compute CLIP image embeddings")
    args = parser.parse_args()

    setup_logging()
    ensure_project_structure()
    train_path = PATHS.data_processed / "train_clean.parquet"
    test_path = PATHS.data_processed / "test_clean.parquet"
    if train_path.exists() and test_path.exists():
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        build_features(
            train_df,
            test_df,
            compute_text_embeddings=not args.skip_text_embeddings,
            compute_image_embeddings=args.image_embeddings,
        )
    else:
        logging.error("Preprocessed parquet files missing. Run data_preprocessing.py first.")
