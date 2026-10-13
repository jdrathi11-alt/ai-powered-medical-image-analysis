"""Project paths and reproducible runtime settings."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Default settings shared by training, evaluation, and prediction."""

    image_size: int = 64
    batch_size: int = 32
    epochs: int = 20
    seed: int = 42
    data_root: Path = PROJECT_ROOT / "data" / "raw"
    model_path: Path = PROJECT_ROOT / "models" / "pneumonia_cnn.keras"
    output_dir: Path = PROJECT_ROOT / "outputs"

    def create_directories(self) -> None:
        """Create writable project directories without creating downloaded data."""
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
