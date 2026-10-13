"""Command-line entry point for the complete educational workflow.

Examples:
    python main.py --mode train --epochs 10
    python main.py --mode evaluate
    python main.py --mode predict --image path/to/xray.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import Settings
from src.data import load_train_validation_test
from src.evaluation import evaluate_model
from src.prediction import predict_image
from src.training import train_model
from src.visualization import save_sample_grid


def _positive_int(value: str) -> int:
    """Parse a strictly positive CLI integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI-powered medical image analysis demo")
    parser.add_argument(
        "--mode",
        choices=("train", "evaluate", "predict"),
        default="train",
        help="workflow step to run",
    )
    parser.add_argument("--image", type=Path, help="image path used with --mode predict")
    parser.add_argument("--epochs", type=_positive_int, default=None)
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument(
        "--max-samples",
        type=_positive_int,
        default=None,
        help="optional per-split cap for a quick smoke run",
    )
    parser.add_argument("--no-download", action="store_true")
    return parser


def _load_model(settings: Settings):
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is not installed. Run `python -m pip install -r requirements.txt`."
        ) from error
    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {settings.model_path}. Run `python main.py --mode train` first."
        )
    return tf.keras.models.load_model(settings.model_path)


def run(args: argparse.Namespace) -> int:
    settings = Settings(
        epochs=args.epochs or Settings.epochs,
        batch_size=args.batch_size or Settings.batch_size,
    )
    settings.create_directories()

    if args.mode == "predict":
        if args.image is None:
            raise ValueError("--image is required when --mode predict is used")
        result = predict_image(
            _load_model(settings),
            args.image,
            image_size=settings.image_size,
            output_path=settings.output_dir / "single_prediction.png",
        )
        print(json.dumps(result, indent=2))
        return 0

    splits = load_train_validation_test(
        data_root=settings.data_root,
        image_size=settings.image_size,
        max_samples=args.max_samples,
        seed=settings.seed,
        download=not args.no_download,
    )
    train_images, train_labels = splits["train"]
    val_images, val_labels = splits["val"]
    test_images, test_labels = splits["test"]

    save_sample_grid(
        train_images,
        train_labels,
        settings.output_dir / "sample_grid.png",
        seed=settings.seed,
    )

    if args.mode == "train":
        model, history = train_model(
            train_images=train_images,
            train_labels=train_labels,
            val_images=val_images,
            val_labels=val_labels,
            model_path=settings.model_path,
            history_path=settings.output_dir / "history.json",
            image_size=settings.image_size,
            batch_size=settings.batch_size,
            epochs=settings.epochs,
            seed=settings.seed,
        )
        metrics = evaluate_model(
            model,
            test_images,
            test_labels,
            settings.output_dir,
            history=history,
        )
        print(json.dumps(metrics, indent=2))
        return 0

    metrics = evaluate_model(
        _load_model(settings),
        test_images,
        test_labels,
        settings.output_dir,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
