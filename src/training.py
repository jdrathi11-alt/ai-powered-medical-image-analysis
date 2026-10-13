"""Model training workflow and persisted training history."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .model import build_model


def _class_weights(labels: np.ndarray) -> dict[int, float]:
    """Calculate balanced weights without leaking validation/test labels."""
    values, counts = np.unique(labels.astype(int), return_counts=True)
    total = counts.sum()
    number_of_classes = len(values)
    return {
        int(value): float(total / (number_of_classes * count))
        for value, count in zip(values, counts)
    }


def train_model(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    model_path: Path,
    history_path: Path,
    image_size: int = 64,
    batch_size: int = 32,
    epochs: int = 20,
    seed: int = 42,
) -> tuple[Any, dict[str, list[float]]]:
    """Train, checkpoint, and return the best validation model."""
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is not installed. Run `python -m pip install -r requirements.txt`."
        ) from error

    tf.keras.utils.set_random_seed(seed)
    model = build_model(image_size=image_size)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=_class_weights(train_labels),
        callbacks=callbacks,
        verbose=1,
    )
    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    history_path.write_text(json.dumps(history_dict, indent=2), encoding="utf-8")
    return model, history_dict
