# Project Specification — AI-Powered Medical Image Analysis System

## Objective

Build a reproducible, beginner-friendly medical-image classification system that
uses public data to simulate an AI assistant for screening support. The first
vertical slice classifies a standardized chest X-ray image as `normal` or
`pneumonia` using the official PneumoniaMNIST dataset.

This is an educational proof-of-work project. It is **not** a medical device,
diagnostic service, triage system, or substitute for a qualified clinician.

## User-visible outcome

An engineer can:

1. Create a clean Python environment.
2. Download the official dataset with one command.
3. Train a compact CNN with a held-out validation split.
4. Evaluate once on the official test split.
5. Open saved plots and metrics demonstrating the experiment.
6. Run inference on a local image and receive a clearly labeled educational
   prediction artifact.

## Dataset contract

- Dataset: MedMNIST PneumoniaMNIST, 64×64 release when available.
- Source: official MedMNIST distribution via Zenodo.
- Task: binary classification.
- Labels: `0 = normal`, `1 = pneumonia`.
- Splits: official train/validation/test splits; the test split is not used for
  model fitting or checkpoint selection.
- License: CC BY 4.0, subject to the dataset's terms and source citations.

## Technical stack

- Python 3.11 or 3.12 recommended.
- NumPy and OpenCV for image arrays and preprocessing.
- TensorFlow/Keras for the CNN and training callbacks.
- scikit-learn for classification metrics.
- Matplotlib for reproducible plots.
- MedMNIST for official data access.

## Project structure

```text
main.py              CLI orchestration
src/                 reusable data, model, training, evaluation, and inference modules
tests/               unit tests for stable project behavior
docs/                project guide, dataset notes, and demo checklist
data/raw/            downloaded dataset files (ignored by Git)
models/              trained Keras files (ignored by Git)
outputs/             metrics and figures (ignored by Git)
```

## Commands

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py --mode train --epochs 20
python main.py --mode evaluate
python main.py --mode predict --image path/to/image.png
```

For a fast local smoke run after dependencies are installed:

```bash
python main.py --mode train --max-samples 64 --epochs 1 --batch-size 16
```

## Code style

- Use type hints for public functions.
- Keep filesystem paths in `pathlib.Path` objects.
- Import TensorFlow, scikit-learn, and other heavy dependencies lazily in the
  function that needs them so label tests can run without the ML stack.
- Keep downloaded data, private inputs, weights, and generated plots out of Git.
- Prefer explicit failure messages over silently using the wrong file or split.

## Testing strategy

- Unit tests: label mapping and preprocessing shape/range contracts.
- Smoke test: one-epoch training on a small deterministic subset after installing
  the ML dependencies.
- Manual acceptance: inspect `outputs/metrics.json`, `sample_grid.png`,
  `training_curves.png`, `confusion_matrix.png`, and
  `prediction_examples.png`.

## Boundaries

- **Always:** preserve the official split, record the dataset and code version,
  use a deterministic seed, make safety limitations visible, and test before
  publishing.
- **Ask first:** adding patient-identifiable data, connecting to hospital
  systems, publishing model weights, changing the dataset, or presenting the
  model as clinically validated.
- **Never:** commit credentials, real patient records, downloaded datasets,
  unsupported accuracy claims, or language that implies clinical diagnosis.

## Success criteria

- [ ] Fresh environment installs from `requirements.txt` on a supported Python
      version.
- [ ] Dataset download is reproducible and stored outside Git.
- [ ] Training writes a `.keras` model and `history.json`.
- [ ] Evaluation writes metrics and at least three evidence plots.
- [ ] Prediction rejects missing/invalid files with a clear error.
- [ ] README explains installation, usage, dataset provenance, limitations,
      results recording, and GitHub proof strategy.
- [ ] No measured accuracy is invented before the training command is run.
