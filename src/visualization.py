"""Saved figures used as evidence for the GitHub project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .labels import CLASS_NAMES, decode_label


def _matplotlib():
    import matplotlib.pyplot as plt

    return plt


def save_sample_grid(
    images: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    count: int = 12,
    seed: int = 42,
) -> None:
    """Save a labeled sample grid showing the input distribution."""
    plt = _matplotlib()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(images), size=min(count, len(images)), replace=False)
    columns = 4
    rows = int(np.ceil(len(indices) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, index in zip(axes, indices):
        axis.imshow(images[index].squeeze(), cmap="gray")
        axis.set_title(decode_label(int(labels[index])))
        axis.axis("off")
    for axis in axes[len(indices) :]:
        axis.axis("off")
    figure.suptitle("PneumoniaMNIST sample images", fontsize=14)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    """Save loss, accuracy, and AUC curves when those metrics exist."""
    plt = _matplotlib()
    figure, axes = plt.subplots(1, 3, figsize=(16, 4))
    metric_groups = (
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("auc", "AUC"),
    )
    for axis, (metric, title) in zip(axes, metric_groups):
        if metric in history:
            axis.plot(history[metric], label="train")
        if f"val_{metric}" in history:
            axis.plot(history[f"val_{metric}"], label="validation")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Training history")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> None:
    """Save an annotated confusion matrix."""
    plt = _matplotlib()
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix",
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                int(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_prediction_grid(
    images: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
    count: int = 12,
) -> None:
    """Save examples with true label, predicted label, and confidence."""
    plt = _matplotlib()
    count = min(count, len(images))
    columns = 4
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(13, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for index, axis in enumerate(axes):
        if index >= count:
            axis.axis("off")
            continue
        true_name = decode_label(int(true_labels[index]))
        prediction = int(predicted_labels[index])
        predicted_name = decode_label(prediction)
        probability = float(probabilities[index])
        confidence = probability if prediction == 1 else 1.0 - probability
        color = "green" if prediction == int(true_labels[index]) else "red"
        axis.imshow(images[index].squeeze(), cmap="gray")
        axis.set_title(
            f"true: {true_name}\npred: {predicted_name} ({confidence:.1%})",
            color=color,
        )
        axis.axis("off")
    figure.suptitle("Prediction examples: green = correct, red = error")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_single_prediction(
    image: np.ndarray,
    predicted_label: int,
    probability: float,
    output_path: Path,
) -> None:
    """Save a single-image prediction card for a demo screenshot."""
    plt = _matplotlib()
    prediction = decode_label(predicted_label)
    confidence = probability if predicted_label == 1 else 1.0 - probability
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(image.squeeze(), cmap="gray")
    axis.set_title(
        f"Educational model output\nPrediction: {prediction} | confidence: {confidence:.1%}",
        fontsize=12,
    )
    axis.axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
