"""Medical-image preprocessing utilities.

The project uses grayscale chest X-rays. Pixel values are converted to float32,
scaled to [0, 1], resized with OpenCV, and given an explicit channel dimension.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _resize_with_opencv(image: np.ndarray, image_size: int) -> np.ndarray:
    """Resize one image using OpenCV when installed, with a Pillow fallback."""
    try:
        import cv2

        return cv2.resize(
            image,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
    except ImportError:
        # The fallback keeps lightweight unit tests usable; normal project runs
        # install opencv-python-headless from requirements.txt.
        from PIL import Image

        pil_image = Image.fromarray(image.astype(np.uint8))
        return np.asarray(
            pil_image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        )


def preprocess_images(images: Any, image_size: int = 64) -> np.ndarray:
    """Return images with shape ``(N, image_size, image_size, 1)`` in [0, 1]."""
    array = np.asarray(images)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(
            f"Expected grayscale image batch with shape (N,H,W), got {array.shape}"
        )

    processed = np.empty((array.shape[0], image_size, image_size, 1), dtype=np.float32)
    for index, image in enumerate(array):
        resized = _resize_with_opencv(image, image_size)
        processed[index, ..., 0] = resized.astype(np.float32) / 255.0
    return processed


def preprocess_single_image(image: Any, image_size: int = 64) -> np.ndarray:
    """Preprocess one grayscale image and return a batch of one."""
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected one grayscale image with shape (H,W), got {array.shape}")
    return preprocess_images(array[None, ...], image_size=image_size)
