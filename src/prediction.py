"""Inference on a user-provided X-ray image."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .labels import decode_label
from .preprocessing import preprocess_single_image
from .visualization import save_single_prediction


def predict_image(
    model: Any,
    image_path: Path,
    image_size: int = 64,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Load one image, predict, and optionally save a prediction card."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is not installed. Run `python -m pip install -r requirements.txt`."
        ) from error

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    batch = preprocess_single_image(image, image_size=image_size)
    probability = float(np.asarray(model.predict(batch, verbose=0)).reshape(-1)[0])
    predicted_label = int(probability >= 0.5)
    confidence = probability if predicted_label == 1 else 1.0 - probability
    result = {
        "image": str(image_path),
        "prediction": decode_label(predicted_label),
        "prediction_id": predicted_label,
        "pneumonia_probability": probability,
        "confidence": confidence,
        "clinical_use": False,
    }
    if output_path is not None:
        save_single_prediction(batch[0], predicted_label, probability, output_path)
    return result
