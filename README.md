# AI-Powered Medical Image Analysis System

> **Educational proof-of-work project:** this repository demonstrates how a
> machine-learning pipeline can assist medical-image review on public data. It
> is **not a medical device, clinical decision-support system, diagnostic tool,
> or substitute for a qualified healthcare professional.** Do not use it for
> patient care.

## A. Project explanation

### What is AI-powered medical image analysis?

In simple language, it means teaching a computer to look at a medical image and
produce a useful, carefully scoped prediction. For example, a model can learn
patterns in labeled chest X-rays and return an educational estimate of whether
an image resembles the `normal` or `pneumonia` examples in its training data.

In technical language, it is a computer-vision and machine-learning workflow:
images are converted to tensors, normalized, passed through learned convolution
filters, and mapped to a class probability. The model is evaluated on images it
did not use during training. In an industry system, that probability would be
combined with quality checks, calibration, human review, audit logs, privacy
controls, validation studies, and regulatory controls.

### Problems it can help address

AI may help healthcare teams with:

- prioritizing cases for review when image volumes are high;
- detecting patterns that deserve a second look;
- reducing repetitive manual screening work;
- making measurements more consistent;
- supporting education and quality-assurance workflows; and
- creating structured metadata for research.

It does **not** eliminate expert review. Models can fail because of data shift,
poor image quality, demographic imbalance, artifacts, shortcuts, or labels that
do not capture the full clinical situation.

### Why organizations explore this use case

| Environment | Practical interest | What a real system would additionally require |
|---|---|---|
| Hospitals | Worklist prioritization and second-look support | PACS/RIS integration, security, monitoring, validation, clinical governance |
| Diagnostic labs | Consistent pre-screening and quality checks | Device/site validation, traceability, fail-safe review |
| Radiology centers | Throughput support and reporting assistance | Radiologist-in-the-loop UX, calibrated uncertainty, auditability |
| Health-tech companies | Research products and workflow automation | Privacy, consent, security, regulatory strategy, post-deployment monitoring |

### Example tasks

- **Classification:** normal vs pneumonia, benign vs malignant, or disease grade.
- **Detection:** locate findings with bounding boxes.
- **Segmentation:** outline an organ, lesion, or region of interest.
- **Image quality assessment:** flag motion, underexposure, or missing anatomy.
- **Triage support:** rank cases for human review; never silently diagnose.

This repository implements classification first because it gives a student a
complete end-to-end system without pretending to solve detection or
segmentation.

### End-to-end workflow

```text
Public dataset
    ↓
Download + provenance record
    ↓
Train / validation / test split
    ↓
Preprocess: grayscale → resize → float32 [0, 1] → channel dimension
    ↓
CNN feature extraction: convolution → normalization → pooling
    ↓
Training: binary cross-entropy + validation AUC checkpoint
    ↓
Prediction: pneumonia probability + thresholded educational label
    ↓
Evaluation: accuracy, precision, recall, AUC, confusion matrix
    ↓
Visualization: sample grid, curves, errors, single-image demo card
    ↓
Documentation: limitations, reproducibility, GitHub proof
```

## B. Technology stack options

### Option A — easiest: compact CNN baseline **(selected)

| Category | Choice |
|---|---|
| Tools | Python, NumPy, OpenCV, Matplotlib, TensorFlow/Keras, scikit-learn |
| Model | Small 2D CNN with augmentation and dropout |
| Dataset | MedMNIST PneumoniaMNIST, 64×64 grayscale images |
| Difficulty | Beginner-friendly |
| GPU | Not required; CPU is sufficient for a small experiment |
| Output | Binary probability, metrics, confusion matrix, plots, prediction card |

Why choose it: every part is visible. You can explain preprocessing, filters,
loss, validation, overfitting, thresholding, and error analysis in an interview.

### Option B — intermediate: transfer learning

| Category | Choice |
|---|---|
| Tools | Python, OpenCV, TensorFlow/Keras Applications, scikit-learn |
| Model | MobileNetV3Small or ResNet50 with a custom binary head |
| Dataset | PneumoniaMNIST 224×224 or another public chest X-ray dataset |
| Difficulty | Intermediate |
| GPU | Helpful; CPU works but is slower |
| Output | Stronger baseline, feature extractor comparison, model-size discussion |

Tradeoff: transfer learning can improve results, but it adds preprocessing,
input-channel, image-size, fine-tuning, and model-license discussion.

### Option C — advanced: detection/segmentation and deployment

| Category | Choice |
|---|---|
| Tools | PyTorch/TensorFlow, MONAI, OpenCV, DICOM tooling, FastAPI, Docker, MLflow |
| Model | U-Net, YOLO-style detector, or 3D CNN with explainability |
| Dataset | Public CT/X-ray/ultrasound dataset with masks or bounding boxes |
| Difficulty | Advanced |
| GPU | Strongly recommended |
| Output | Regions of interest, masks/boxes, API or demo application, experiment tracking |

Tradeoff: this is closer to production engineering but much harder to validate,
explain, and complete responsibly as a first project.

## C. Selected approach

### Project decision

This repository uses **Option A**:

- **Task:** normal-vs-pneumonia image classification.
- **Dataset:** official MedMNIST PneumoniaMNIST, 64×64 release.
- **Model:** compact Keras CNN with light augmentation, batch normalization,
  global average pooling, dropout, and a sigmoid output.
- **Metrics:** loss, accuracy, AUC, precision, recall, confusion matrix,
  classification report.
- **Reproducibility:** official splits, seed 42, saved history, saved model,
  explicit data and model paths.
- **Safety:** results are educational estimates only; no clinical claims.

### Dataset provenance

MedMNIST describes PneumoniaMNIST as a binary classification dataset based on
5,856 pediatric chest X-ray images and provides official train/validation/test
splits. It is distributed under CC BY 4.0 and the project says the dataset is
not intended for clinical use. Use the official distribution links and cite the
MedMNIST paper in any public write-up.

- Official project: <https://medmnist.com/>
- Official repository: <https://github.com/MedMNIST/MedMNIST>
- Official Zenodo distribution: <https://doi.org/10.5281/zenodo.10519652>
- MedMNIST paper: <https://doi.org/10.1038/s41597-022-01721-8>

## D. Architecture

### Text-based block diagram

```text
                  ┌──────────────────────────┐
                  │ MedMNIST public dataset  │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │ Official train/val/test  │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │ OpenCV preprocessing     │
                  │ resize + normalize       │
                  └────────────┬─────────────┘
                               │ (N, 64, 64, 1)
                  ┌────────────▼─────────────┐
                  │ CNN feature extractor    │
                  │ Conv → BN → Pool × 3     │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │ Classification head      │
                  │ GAP → Dense → Dropout    │
                  │ Sigmoid probability      │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │ Metrics + visual outputs │
                  │ human-reviewed demo      │
                  └──────────────────────────┘
```

### Modules

| Module | Responsibility |
|---|---|
| `src/config.py` | Central paths and default settings |
| `src/data.py` | Download and load official splits; deterministic subset option |
| `src/preprocessing.py` | Resize, normalize, and shape-check images |
| `src/model.py` | Build and compile the CNN |
| `src/training.py` | Class weights, callbacks, checkpoint, history |
| `src/evaluation.py` | Test metrics and evidence artifacts |
| `src/prediction.py` | Read a local image and create an educational output |
| `src/visualization.py` | Sample grids, curves, matrices, and prediction cards |
| `main.py` | User-facing CLI orchestration |

### Data flow

1. The loader downloads the official PneumoniaMNIST `.npz` file into
   `data/raw/` and verifies its MD5 checksum.
2. `src/data.py` reads only one named split at a time.
3. `src/preprocessing.py` resizes each grayscale image and scales pixels to
   `[0, 1]`.
4. Training uses train data and selects the best checkpoint using validation
   AUC.
5. Evaluation runs once on the held-out test set and writes JSON plus PNG files.
6. Prediction loads a saved model and returns a probability and label. It does
   not know a patient's history and must not be used as a diagnosis.

## E. Folder structure

```text
AI-Powered-Medical-Image-Analysis/
├── data/
│   ├── raw/.gitkeep          # dataset downloads; ignored by Git
│   └── processed/.gitkeep    # optional future exports; ignored by Git
├── docs/
│   ├── PROJECT_SPEC.md       # scope and acceptance criteria
│   └── BEGINNER_GUIDE.md     # detailed phases and GitHub proof plan
├── images/
│   └── generated/.gitkeep    # optional screenshots/assets; ignored by Git
├── models/.gitkeep           # trained .keras files; ignored by Git
├── notebooks/                # optional exploration notebooks
├── outputs/.gitkeep          # metrics and plots; ignored by Git
├── src/
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── labels.py
│   ├── model.py
│   ├── prediction.py
│   ├── preprocessing.py
│   ├── training.py
│   └── visualization.py
├── tests/
│   ├── test_labels.py
│   └── test_preprocessing.py
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

Generated files are intentionally ignored. For GitHub proof, export a small
set of screenshots into `images/` only after reviewing them for privacy and
correctness. Do not commit the downloaded dataset or real patient images.

## F. Installation and environment setup

### Recommended Python version

Use **Python 3.11 or 3.12** for the smoothest TensorFlow installation. This
workspace currently has Python 3.14 and 3.13 available; TensorFlow wheel
availability varies by operating system and Python version, so use a supported
Python interpreter if installation fails.

### Windows PowerShell

```powershell
cd "F:\Projects AI\AI-Powered Medical Image Analysis"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

If PowerShell blocks activation, either run the command in Command Prompt:

```bat
.venv\Scripts\activate.bat
```

or use the current PowerShell session's approved execution policy. Do not
disable system security broadly just to activate a virtual environment.

### macOS/Linux

```bash
cd "/path/to/AI-Powered Medical Image Analysis"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### `requirements.txt`

The repository pins compatible major-version ranges for TensorFlow, NumPy,
OpenCV, Matplotlib, scikit-learn, Pillow, and pytest. A fresh setup
must install from that file rather than relying on globally installed packages.

The requirements file does not install the optional `medmnist` Python wrapper:
this project reads the official MedMNIST `.npz` archive directly from Zenodo,
as documented by the MedMNIST project, so a TensorFlow/Keras beginner does not
need a large PyTorch/torchvision dependency just to read NumPy data.

### CPU/GPU guidance

The selected baseline is small enough for CPU experimentation, especially with
`--max-samples`. A GPU is useful for full runs, but it is not a requirement for
learning or producing the proof artifacts. Do not claim speed or accuracy until
you record the hardware, dataset release, seed, epochs, and actual output.

## G. Complete implementation

The source code is split into small files so a beginner can read one concept at
a time. Each file is executable as part of `main.py`.

### `src/labels.py`

Defines the stable binary label contract:

```python
CLASS_NAMES = ("normal", "pneumonia")
```

`decode_label()` is used for human-readable reports and `encode_label()` helps
tests and future UI code reject unsupported labels instead of silently guessing.

### `src/config.py`

Uses `pathlib.Path` to keep project paths portable. The `Settings` dataclass
contains image size, batch size, epochs, seed, data location, model location,
and output location. It creates only the directories the workflow needs.

### `src/data.py`

Downloads the official MedMNIST NumPy archive from Zenodo, verifies its MD5
checksum, and reads one explicit split at a time. It offers an optional
deterministic `--max-samples` cap for smoke tests. The cap is not a replacement
for a full experiment; it is a fast way to verify that wiring works.

### `src/preprocessing.py`

Checks that data are a grayscale batch, resizes with OpenCV, converts to
`float32`, divides by 255, and returns channel-last tensors with shape
`(N, image_size, image_size, 1)`. This contract is tested in `tests/`.

### `src/model.py`

Builds a compact Keras `Sequential` CNN:

```text
input → augmentation → Conv2D/BatchNorm/Pool × 3
      → GlobalAveragePooling → Dropout → Dense → sigmoid
```

The output is one number in `[0, 1]`, interpreted as the model's pneumonia
probability under the educational dataset definition.

### `src/training.py`

Uses train labels for balanced class weights and saves the best `.keras` model
according to validation AUC. Early stopping restores the best weights. A
JSON-serializable history is written to `outputs/history.json`.

### `src/evaluation.py`

Runs Keras metrics and scikit-learn metrics on the held-out test split, then
writes `outputs/metrics.json`, `confusion_matrix.png`, and
`prediction_examples.png`. It also writes training curves when history is
available.

### `src/prediction.py`

Reads one local grayscale image with OpenCV, applies the same preprocessing,
loads the saved model through the CLI, and returns a JSON-compatible result.
The output includes `clinical_use: false` to keep the safety boundary visible.

### `src/visualization.py`

Creates screenshot-ready plots: input sample grid, training curves, confusion
matrix, correct/error prediction grid, and a single-image prediction card.

### `main.py`

The only file a beginner needs to run. It has three modes:

```bash
python main.py --mode train
python main.py --mode evaluate
python main.py --mode predict --image path/to/image.png
```

Useful flags:

```bash
python main.py --mode train --epochs 5 --batch-size 32
python main.py --mode train --max-samples 128 --epochs 1 --batch-size 16
python main.py --mode evaluate --no-download
```

`--no-download` is useful on an offline machine after the dataset has already
been placed in the MedMNIST root. It should not be used before the file exists.

## H. Virtual simulation implementation

### What is being simulated?

There is no hospital integration in this project. The simulation has three
parts:

1. **Data simulation:** a public, de-identified research dataset stands in for
   an image archive.
2. **Model simulation:** the trained CNN stands in for an automated screening
   assistant that produces a probability, not a final diagnosis.
3. **Workflow simulation:** saved plots and a prediction card stand in for the
   review artifacts an engineer might show in a prototype.

The labels do not contain the complete clinical truth, and a benchmark result
does not establish clinical utility. The simulation is valuable for learning
the engineering workflow, not for treating patients.

### Step-by-step virtual workflow

1. Install the environment.
2. Run a unit test and one-epoch small-subset smoke run.
3. Download the official dataset into ignored local storage.
4. Generate `sample_grid.png` to inspect class examples.
5. Train using only train data; use validation data for checkpoint selection.
6. Inspect `history.json` and `training_curves.png` for overfitting.
7. Evaluate once on test data.
8. Inspect confusion matrix, precision, recall, and errors.
9. Pick a local sample image only for an educational demo.
10. Run single-image prediction and save `single_prediction.png`.
11. Capture the terminal command, output list, and plots.
12. Document dataset provenance, hardware, seed, run configuration, and
    limitations in the GitHub README.

### Proof artifacts to generate

| Artifact | What it proves |
|---|---|
| `sample_grid.png` | Data can be loaded and labels are visible |
| `history.json` | Training metrics were actually recorded |
| `training_curves.png` | Training/validation behavior can be inspected |
| `metrics.json` | Test metrics and confusion counts are persisted |
| `confusion_matrix.png` | Class-specific errors are visible |
| `prediction_examples.png` | Predictions are compared with true labels |
| `single_prediction.png` | The inference path works on one local image |
| terminal screenshot | The run is reproducible from documented commands |

Never replace a missing artifact with a made-up screenshot or invented metric.

## I. Execution plan and expected outputs

### Phase 1 — Setup

**Do:** create `.venv`, install requirements, and run unit tests.

**Why:** isolates dependencies and catches import/path problems before any
download or training.

**Expected:** two test modules pass; the environment has the required packages.

**Common mistakes:** using a different Python interpreter for `pip` and
`python`, installing into the global environment, or running from the wrong
folder.

**Verify:**

```bash
python --version
python -m pip show tensorflow numpy opencv-python-headless
python -m unittest discover -s tests -v
```

### Phase 2 — Dataset collection

**Do:** let `main.py` download the official MedMNIST NumPy archive.

**Why:** keeps data provenance and official split definitions reproducible.

**Expected:** a PneumoniaMNIST `.npz` file appears under the local data root;
it remains ignored by Git.

**Common mistakes:** downloading random images from a search engine, mixing
train and test data, or committing the dataset into the repository.

**Verify:**

```bash
python main.py --mode train --max-samples 64 --epochs 1 --batch-size 16
```

### Phase 3 — Image preprocessing

**Do:** inspect the sample grid and preprocessing tests.

**Why:** consistent shape and normalization are prerequisites for the model.

**Expected:** `outputs/sample_grid.png` and tensors shaped `(N, 64, 64, 1)`.

**Common mistakes:** using color assumptions for grayscale data, normalizing
train and test differently, or resizing after the model receives the data.

**Verify:** `python -m unittest discover -s tests -v` and inspect the PNG.

### Phase 4 — Model building

**Do:** read `src/model.py` and run the smoke training command.

**Why:** the model must compile before a long run starts.

**Expected:** Keras prints a model with a single sigmoid output and no shape
error.

**Common mistakes:** using a two-unit softmax with binary loss incorrectly,
forgetting the channel dimension, or adding a large model before establishing a
baseline.

**Verify:** the smoke run reaches the first epoch and writes a `.keras` file.

### Phase 5 — Training

**Do:** run the complete training workflow.

```bash
python main.py --mode train --epochs 20 --batch-size 32
```

**Why:** learns parameters from train data and selects a checkpoint using only
validation AUC.

**Expected:** `models/pneumonia_cnn.keras`, `outputs/history.json`, and curves.

**Common mistakes:** increasing epochs to chase a metric without inspecting
validation behavior, training on test data, or reporting a smoke-run metric as
the final experiment.

**Verify:** confirm the saved model timestamp and inspect `history.json`.

### Phase 6 — Evaluation

**Do:** use the official test split once the model is frozen.

```bash
python main.py --mode evaluate --no-download
```

**Why:** measures generalization on data excluded from fitting and checkpoint
selection.

**Expected:** `metrics.json`, confusion matrix, and prediction examples.

**Common mistakes:** tuning the threshold on the test set, reporting accuracy
without class-specific recall/precision, or hiding class imbalance.

**Verify:** record the exact JSON values with the dataset release, seed, and
hardware in the experiment notes. This README deliberately contains no made-up
result numbers.

The completed small wiring check is recorded in
[`docs/SMOKE_RUN.md`](docs/SMOKE_RUN.md). It is explicitly labeled as a smoke
run, not a final benchmark.

### Phase 7 — Prediction

**Do:** provide a local PNG/JPG image.

```bash
python main.py --mode predict --image path/to/xray.png
```

**Why:** demonstrates the user-facing inference path.

**Expected:** terminal JSON and `outputs/single_prediction.png`.

**Common mistakes:** using non-X-ray images, interpreting probability as
certainty, or calling the result a diagnosis.

**Verify:** the output image and terminal JSON refer to the same input.

### Phase 8 — Visualization

**Do:** review all PNGs, including false positives and false negatives.

**Why:** a single score cannot explain model behavior or expose errors.

**Expected:** a coherent visual story from input to result.

**Common mistakes:** showing only successful predictions, cropping out labels,
or using plots without axis titles and data provenance.

**Verify:** select 3–5 reviewed images for the GitHub README or a release note.

### Phase 9 — GitHub publishing

**Do:** commit source, tests, docs, and a clean README; keep data and weights
ignored.

**Why:** reproducible code is stronger proof than a large binary file.

**Expected:** a visitor can clone the repo and reproduce the workflow.

**Common mistakes:** committing `.venv`, `.env`, dataset files, secrets, or
unsupported claims.

**Verify:** run `git status --ignored`, inspect `git diff --cached`, and search
the staged diff for secret-like strings before pushing.

### Phase 10 — Final output

**Do:** package a small demo narrative: problem, data, architecture, command,
metrics, errors, limitations, and next steps.

**Why:** interviewers assess reasoning and engineering discipline, not just a
number.

**Expected:** a polished repo plus a 60–90 second screen recording or a set of
reviewed screenshots.

**Common mistakes:** claiming clinical readiness, omitting dataset citations,
or presenting generated assets without explaining how to reproduce them.

**Verify:** use the proof checklist below and ask a peer to follow the README
from a clean environment.

### Sample output shape

The exact values depend on the run and must be captured from `metrics.json`. A
prediction response has this general form:

```json
{
  "image": "path/to/xray.png",
  "prediction": "normal",
  "prediction_id": 0,
  "pneumonia_probability": 0.18,
  "confidence": 0.82,
  "clinical_use": false
}
```

The values above are **format examples only**, not measured results.

## J. GitHub upload strategy

### Recommended repository identity

- **Repository name:** `ai-powered-medical-image-analysis`
- **Description:** `Educational chest X-ray classification pipeline using public MedMNIST data, TensorFlow/Keras, reproducible evaluation, and visual proof artifacts.`
- **Suggested topics:** `medical-imaging`, `computer-vision`, `deep-learning`,
  `tensorflow`, `keras`, `python`, `pneumonia-detection`, `medmnist`,
  `machine-learning`, `portfolio-project`

### Create and push

After reviewing the files and installing Git:

```bash
git init
git branch -M main
git add .
git status
git diff --cached
git commit -m "chore: initialize medical image analysis project"
git remote add origin https://github.com/YOUR_USERNAME/ai-powered-medical-image-analysis.git
git push -u origin main
```

Use your actual GitHub URL. Never place a GitHub token or password in a file or
command that will be saved in shell history. If Git asks for credentials, use
GitHub's supported authentication flow.

### Professional commit strategy

Use small commits that each prove one idea:

```text
chore: initialize reproducible Python project
docs: add scope, dataset provenance, and safety boundary
feat: add MedMNIST data loading and preprocessing
feat: add compact CNN training workflow
feat: add test evaluation and visual evidence artifacts
test: cover label and preprocessing contracts
docs: add reproducibility and GitHub proof checklist
```

### Images and videos

- Put reviewed, privacy-safe screenshots in `images/`.
- Keep large videos in a GitHub release, Git LFS, or an external portfolio link.
- Do not upload the raw medical dataset or any real patient image.
- Add captions: command, run date, dataset release, seed, and what the image
  proves.
- Link to a short demo video from the README only after checking the link.

## K. README content checklist

This file already contains the core README sections requested for a portfolio:

- overview and safety statement;
- problem statement and industry relevance;
- beginner/intermediate/advanced options;
- selected architecture and data flow;
- folder structure;
- Windows and macOS/Linux installation;
- commands for train/evaluate/predict;
- file-by-file implementation explanation;
- public dataset and licensing links;
- simulation workflow;
- evaluation and visualization outputs;
- GitHub upload strategy;
- seven-day commit and proof plan;
- final limitations and next steps.

When you have real results, add a small table like this using values copied
from `outputs/metrics.json`:

| Run | Dataset | Image size | Epochs | Test accuracy | Test AUC | Recall | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `YYYY-MM-DD-seed42` | PneumoniaMNIST | 64 | 20 | `paste` | `paste` | `paste` | CPU/GPU + early stopping |

Do not use placeholder values in a public results table.

## L. Seven-day GitHub proof plan

### Day 1 — Setup

**Commit:** `chore: initialize reproducible Python project`

**Proof:** Python version, environment install, requirements file, passing unit
tests, folder tree.

### Day 2 — Dataset

**Commit:** `docs: document MedMNIST provenance and data policy`

**Proof:** official dataset link, license citation, label mapping, confirmation
that data are ignored and not committed.

### Day 3 — Preprocessing

**Commit:** `feat: add deterministic image preprocessing`

**Proof:** sample grid, tensor shape/range test, before/after visualization.

### Day 4 — Model

**Commit:** `feat: add compact CNN baseline`

**Proof:** model summary screenshot, architecture diagram, explanation of the
sigmoid output and binary cross-entropy.

### Day 5 — Training

**Commit:** `feat: add checkpointed model training`

**Proof:** terminal training log, `history.json`, training/validation curves,
saved best-model path.

### Day 6 — Evaluation

**Commit:** `feat: add holdout evaluation and error visualizations`

**Proof:** metrics JSON, confusion matrix, correct/error grid, discussion of
precision vs recall and limitations.

### Day 7 — Upload

**Commit:** `docs: publish reproducible GitHub demo guide`

**Proof:** public repository, polished README, screenshots, demo video or GIF,
dataset citation, safety statement, and a “how to reproduce” section.

## M. Screenshots, outputs, and demo assets

### Asset list

| Asset | Suggested filename | Location | README usage |
|---|---|---|---|
| Input sample grid | `01-sample-grid.png` | `images/` | “Dataset and labels” |
| Preprocessing view | `02-preprocessing.png` | `images/` | “Preprocessing” |
| Training logs | `03-training-log.png` | `images/` | “Training” |
| Accuracy/loss/AUC | `04-training-curves.png` | `images/` | “Results” |
| Confusion matrix | `05-confusion-matrix.png` | `images/` | “Error analysis” |
| Prediction grid | `06-predictions.png` | `images/` | “Inference” |
| Single prediction card | `07-single-prediction.png` | `images/` | “Demo” |
| GitHub preview | `08-github-repo.png` | `images/` | “Portfolio” |

Generated outputs first appear under `outputs/`. Copy only reviewed and
privacy-safe assets to `images/`; do not copy the entire output directory or
model weights without checking size, license, and reproducibility.

### Suggested README image block after a real run

```markdown
## Demo artifacts

| Input samples | Training curves |
|---|---|
| ![Input sample grid](images/01-sample-grid.png) | ![Training curves](images/04-training-curves.png) |

| Confusion matrix | Prediction examples |
|---|---|
| ![Confusion matrix](images/05-confusion-matrix.png) | ![Prediction examples](images/06-predictions.png) |
```

### Demo narration

In a 60–90 second demo:

1. Show the repository and the safety disclaimer.
2. Show the official dataset source and label definition.
3. Run the smoke command or show the recorded full command.
4. Open the curves and explain validation checkpointing.
5. Open the confusion matrix and point out one error type.
6. Run prediction on an educational sample.
7. End with limitations and next steps.

## N. Limitations and responsible next steps

### Limitations of this student project

- The dataset is standardized and small compared with real clinical archives.
- Pediatric chest X-ray data may not generalize to adults or other sites.
- Labels and images can contain dataset-specific shortcuts and noise.
- The model has no patient history, symptoms, laboratory data, or radiologist
  context.
- A probability is not a calibrated risk estimate unless calibration is tested.
- Accuracy alone is not enough; subgroup performance, sensitivity, specificity,
  calibration, and external validation would matter.
- There is no PACS/RIS integration, access control, audit log, monitoring,
  clinical validation, regulatory clearance, or human-factors study.

### Good advanced extensions

1. Add a MobileNetV3Small transfer-learning comparison while keeping the CNN as
   a baseline.
2. Add calibration curves and threshold analysis for a chosen operating point.
3. Add stratified metrics and confidence intervals where metadata supports it.
4. Add Grad-CAM only as an exploratory visualization, not a proof of reasoning.
5. Add a FastAPI or Streamlit demo that blocks unsupported file types and shows
   the safety warning on every result.
6. Add experiment tracking with run configuration and dataset hash.
7. Use an external public dataset for a carefully described robustness check.
8. Add DICOM handling only with synthetic or public images and strict privacy
   controls.

## Learning outcomes

After completing this project, a student can explain:

- how image tensors move through a CNN;
- why data splits prevent leakage;
- why validation and test data serve different purposes;
- how normalization and resizing affect training;
- how accuracy, precision, recall, AUC, and confusion matrices differ;
- why checkpointing and early stopping help with overfitting;
- how to create reproducible CLI workflows;
- how to separate generated artifacts from source code in Git;
- why a benchmark result is not clinical evidence; and
- how to communicate limitations honestly in a technical interview.

## Sources and citations

- MedMNIST official site: <https://medmnist.com/>
- MedMNIST official repository: <https://github.com/MedMNIST/MedMNIST>
- MedMNIST official Zenodo record: <https://doi.org/10.5281/zenodo.10519652>
- MedMNIST v2 paper: <https://doi.org/10.1038/s41597-022-01721-8>
- Keras `ModelCheckpoint`: <https://keras.io/api/callbacks/model_checkpoint/>
- Keras `EarlyStopping`: <https://keras.io/api/callbacks/early_stopping/>
- TensorFlow installation guidance: <https://www.tensorflow.org/install/pip>

## License and data notice

The project source is provided for educational use. The MedMNIST data are
subject to their own CC BY 4.0 terms and source citations. Review the official
dataset terms before redistribution. Do not include downloaded data in this
repository.
