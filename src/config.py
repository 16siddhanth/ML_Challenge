"""Configuration utilities for the multimodal price prediction project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class PathConfig:
    """Filesystem layout for data, artifacts, and logs."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_raw: Path = field(init=False)
    data_processed: Path = field(init=False)
    models_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    student_resources: Path = field(init=False)
    student_dataset: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_raw", self.project_root / "data" / "raw")
        object.__setattr__(self, "data_processed", self.project_root / "data" / "processed")
        object.__setattr__(self, "models_dir", self.project_root / "models")
        object.__setattr__(self, "outputs_dir", self.project_root / "outputs")
        object.__setattr__(self, "logs_dir", self.project_root / "logs")
        object.__setattr__(self, "reports_dir", self.project_root / "reports")
        object.__setattr__(self, "student_resources", self.project_root / "student_resource")
        object.__setattr__(self, "student_dataset", (self.project_root / "student_resource" / "dataset"))

    @property
    def outputs(self) -> Path:
        """Backward-compatible alias used by older modules."""

        return self.outputs_dir

    def locate_dataset_file(self, filename: str) -> Optional[Path]:
        """Locate dataset file in raw data directory or student resources."""

        candidates = [
            self.data_raw / filename,
            self.student_dataset / filename,
        ]
        for path in candidates:
            if path.exists():
                return path
        return None


@dataclass(frozen=True)
class TrainingConfig:
    """Common training hyperparameters."""

    random_seed: int = 42
    n_splits: int = 5
    validation_size: float = 0.1
    target_column: str = "price"
    stratify_bins: int = 20
    price_clip_min: Optional[float] = None
    price_clip_max: Optional[float] = None
    log_target: bool = True


PATHS = PathConfig()
TRAINING = TrainingConfig()
