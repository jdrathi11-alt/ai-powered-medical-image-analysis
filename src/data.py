"""Download and load official MedMNIST PneumoniaMNIST NumPy files.

The MedMNIST project documents that its data files are standard NumPy ``.npz``
archives and can be used without PyTorch. This module uses the official Zenodo
file directly so the TensorFlow/Keras project does not need to install an
unrelated PyTorch stack just to read the dataset.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

from .preprocessing import preprocess_images


DATASET_NAME = "pneumoniamnist"
DATASET_SIZE = 64
DATASET_URL = (
    "https://zenodo.org/records/10519652/files/"
    "pneumoniamnist_64.npz?download=1"
)
DATASET_MD5 = "8f4eceb4ccffa70c672198ea285246c6"


def _dataset_path(data_root: Path) -> Path:
    return data_root / f"{DATASET_NAME}_{DATASET_SIZE}.npz"


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_dataset(data_root: Path, download: bool) -> Path:
    """Ensure the official 64×64 archive exists and matches its MD5."""
    data_root.mkdir(parents=True, exist_ok=True)
    path = _dataset_path(data_root)
    if path.exists():
        if _md5(path) != DATASET_MD5:
            raise RuntimeError(
                f"Dataset checksum failed for {path}. Delete it and download again."
            )
        return path
    if not download:
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run with downloads enabled first."
        )

    temporary_path = path.with_suffix(".download")
    try:
        urllib.request.urlretrieve(DATASET_URL, temporary_path)
        if _md5(temporary_path) != DATASET_MD5:
            raise RuntimeError(
                "Downloaded dataset checksum failed; the file was not accepted."
            )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path


def _load_medmnist_split(
    split: str,
    data_root: Path,
    image_size: int,
    download: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Load raw image arrays and labels from one official split."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val', or 'test'")

    if image_size != DATASET_SIZE:
        raise ValueError(
            f"This baseline uses the official {DATASET_SIZE}x{DATASET_SIZE} archive."
        )
    path = _ensure_dataset(data_root, download=download)
    with np.load(path) as archive:
        return (
            np.asarray(archive[f"{split}_images"]).copy(),
            np.asarray(archive[f"{split}_labels"]).reshape(-1).copy(),
        )


def _select_subset(
    images: np.ndarray,
    labels: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a deterministic, approximately class-balanced subset."""
    if max_samples is None or max_samples >= len(images):
        return images, labels
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2 when provided")

    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max_samples // len(classes)
    chosen: list[int] = []
    for class_id in classes:
        candidates = np.flatnonzero(labels == class_id)
        count = min(per_class, len(candidates))
        chosen.extend(rng.choice(candidates, size=count, replace=False).tolist())

    remaining = max_samples - len(chosen)
    if remaining > 0:
        available = np.setdiff1d(np.arange(len(images)), np.asarray(chosen))
        chosen.extend(rng.choice(available, size=remaining, replace=False).tolist())

    indices = np.asarray(chosen, dtype=np.int64)
    rng.shuffle(indices)
    return images[indices], labels[indices]


def load_split(
    split: str,
    data_root: Path,
    image_size: int = 64,
    max_samples: int | None = None,
    seed: int = 42,
    download: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load, preprocess, and optionally limit one dataset split."""
    raw_images, labels = _load_medmnist_split(
        split=split,
        data_root=data_root,
        image_size=image_size,
        download=download,
    )
    raw_images, labels = _select_subset(raw_images, labels, max_samples, seed)
    return preprocess_images(raw_images, image_size=image_size), labels.astype(np.float32)


def load_train_validation_test(
    data_root: Path,
    image_size: int = 64,
    max_samples: int | None = None,
    seed: int = 42,
    download: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all official splits, applying the same optional cap to each split."""
    return {
        split: load_split(
            split=split,
            data_root=data_root,
            image_size=image_size,
            max_samples=max_samples,
            seed=seed + offset,
            download=download,
        )
        for offset, split in enumerate(("train", "val", "test"))
    }
