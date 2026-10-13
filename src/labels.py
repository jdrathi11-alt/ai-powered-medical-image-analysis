"""Label definitions and safe conversions for PneumoniaMNIST."""

CLASS_NAMES = ("normal", "pneumonia")


def decode_label(label: int) -> str:
    """Return a human-readable label for a model target."""
    value = int(label)
    if value < 0 or value >= len(CLASS_NAMES):
        raise ValueError(f"Unknown label {value}; expected 0 or 1")
    return CLASS_NAMES[value]


def encode_label(name: str) -> int:
    """Convert a class name into its integer target."""
    normalized = name.strip().lower()
    if normalized not in CLASS_NAMES:
        raise ValueError(f"Unknown class {name!r}; expected one of {CLASS_NAMES}")
    return CLASS_NAMES.index(normalized)
