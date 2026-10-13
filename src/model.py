"""Compact CNN model for a reproducible educational baseline."""

from __future__ import annotations

from typing import Any


def build_model(image_size: int = 64) -> Any:
    """Build and compile a binary image classifier with Keras."""
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is not installed. Run `python -m pip install -r requirements.txt`."
        ) from error

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size, image_size, 1)),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(height_factor=0.05, width_factor=0.05),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="pneumonia_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
