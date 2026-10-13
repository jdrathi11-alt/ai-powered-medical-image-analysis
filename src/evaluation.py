"""Hold-out evaluation and evidence artifact generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .labels import CLASS_NAMES
from .visualization import (
    save_confusion_matrix,
    save_prediction_grid,
    save_training_curves,
)


def evaluate_model(
    model: Any,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    output_dir: Path,
    history: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen test split and save metrics and figures."""
    try:
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            roc_auc_score,
        )
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn is not installed. Run `python -m pip install -r requirements.txt`."
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    keras_metrics = model.evaluate(test_images, test_labels, verbose=0, return_dict=True)
    probabilities = np.asarray(model.predict(test_images, verbose=0)).reshape(-1)
    predicted_labels = (probabilities >= 0.5).astype(int)
    integer_labels = test_labels.astype(int)
    matrix = confusion_matrix(integer_labels, predicted_labels, labels=[0, 1])
    report = classification_report(
        integer_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "dataset": "PneumoniaMNIST",
        "clinical_use": False,
        "test_samples": int(len(test_labels)),
        "keras_metrics": {key: float(value) for key, value in keras_metrics.items()},
        "roc_auc": float(roc_auc_score(integer_labels, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if history:
        save_training_curves(history, output_dir / "training_curves.png")
    save_confusion_matrix(matrix, output_dir / "confusion_matrix.png")
    save_prediction_grid(
        test_images,
        integer_labels,
        predicted_labels,
        probabilities,
        output_dir / "prediction_examples.png",
    )
    return metrics
