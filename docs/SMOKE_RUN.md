# Smoke-run verification record

This record documents the completed wiring check. It is **not** a final model
benchmark and must not be presented as clinical evidence.

## Environment

- OS: Windows
- Python: 3.13.15
- TensorFlow: 2.21.0
- NumPy: 2.5.4
- OpenCV: 4.14.0
- Test runner: pytest 8.4.2 and Python `unittest`
- Hardware: CPU; TensorFlow reported that native Windows GPU support was not
  available in this environment.

## Command

```powershell
.\.venv\Scripts\python.exe main.py --mode train --max-samples 64 --epochs 1 --batch-size 16
```

The smoke run uses 64 examples from each official split, selected
deterministically with seed 42. It confirms the code path, not generalization.

## Actual output summary

The run completed and wrote the model, history, metrics, and visual artifacts.

| Metric | Value |
|---|---:|
| Test samples | 64 |
| Test accuracy | 0.5000 |
| Test AUC | 0.5342 |
| Test precision | 0.0000 |
| Test recall | 0.0000 |
| ROC AUC | 0.5342 |

The confusion matrix was:

```text
[[32, 0],
 [32, 0]]
```

The result is intentionally weak because the model saw only one training epoch
and 64 samples. It is evidence that the workflow executes, not evidence that
the model is useful. Run the full documented experiment before discussing model
quality.

## Separate saved-model checks

These commands also completed successfully after the smoke run:

```powershell
.\.venv\Scripts\python.exe main.py --mode evaluate --max-samples 64 --no-download
.\.venv\Scripts\python.exe main.py --mode predict --image "outputs\demo_input.png"
```

The saved-model prediction returned a probability of approximately `0.4845` for
the local public-data demo image and the educational label `normal`. This is a
format and loading check only; it is not a medical conclusion.
