# Beginner-to-Industry Guide

This guide is the step-by-step companion to the repository. Read it in order
the first time. The source files are the executable truth; this document explains
why each part exists and how to turn it into credible GitHub proof.

## Safety boundary first

This project uses public research data to demonstrate an engineering workflow.
It does not diagnose, rule out disease, triage a real person, or replace a
radiologist. Never upload identifiable patient data. Never connect this demo to
a hospital system. Every result should be described as an educational model
output.

## A. Project explanation

### Simple explanation

Medical images are pictures produced by equipment such as X-ray, CT, MRI,
ultrasound, OCT, or dermatoscopes. AI-powered medical image analysis uses
examples with labels to learn visual patterns. Given a new image, the computer
returns a prediction that can help a qualified person decide what to review.

The computer does not “understand” a patient the way a doctor does. It only
learns statistical relationships in its training data. Therefore the safe use
case is assistance, not autonomous diagnosis.

### Technical explanation

An image is represented as a tensor. Preprocessing standardizes dimensions and
pixel ranges. A convolutional neural network learns filters that respond to
local edges, textures, and higher-level combinations. Pooling or global average
pooling compresses spatial information. A classification head maps extracted
features to a probability. A loss function measures error, and optimization
updates the weights on training images.

The test set is held out until the model and thresholding decisions are fixed.
This makes the final estimate less biased than evaluating on data used during
model selection.

### Healthcare problems and realistic boundaries

AI can potentially support high-volume screening, second-look review, image
quality checks, structured measurements, teaching, and research. Hospitals,
labs, radiology centers, and health-tech teams care about these areas because
they work with large image volumes and need reliable, traceable workflows.

In a real deployment, the model would also need data governance, privacy,
cybersecurity, human factors, calibration, external validation, monitoring,
versioning, incident response, and regulatory review. This portfolio project
implements none of those clinical controls; it demonstrates the ML pipeline.

### Full workflow in plain language

1. Find an authorized public dataset.
2. Understand the image type, labels, license, and official split.
3. Load images without mixing train, validation, and test roles.
4. Resize and normalize every image consistently.
5. Train a model on the train split.
6. Select checkpoints with the validation split.
7. Freeze the selected model.
8. Evaluate once on the test split.
9. Visualize both correct predictions and errors.
10. Save the configuration and artifacts so another person can reproduce it.

## B. Three stack options

### Option A: compact CNN

- **Libraries:** Python, NumPy, OpenCV, TensorFlow/Keras, Matplotlib,
  scikit-learn, MedMNIST.
- **Model:** 2D CNN with light augmentation, batch normalization, dropout, and
  sigmoid output.
- **Data:** 64×64 grayscale PneumoniaMNIST.
- **Difficulty:** beginner.
- **GPU:** optional.
- **Output:** probability, label, metrics, plots, confusion matrix, and saved
  model.

### Option B: transfer learning

- **Libraries:** the Option A stack plus Keras Applications.
- **Model:** MobileNetV3Small or a lightweight ResNet with a custom head.
- **Data:** 128×128 or 224×224 public X-rays.
- **Difficulty:** intermediate.
- **GPU:** helpful.
- **Output:** baseline comparison, model-size and fine-tuning discussion.

### Option C: advanced clinical-style prototype

- **Libraries:** PyTorch or TensorFlow, MONAI, DICOM tooling, FastAPI,
  Docker, MLflow, testing and monitoring tools.
- **Model:** U-Net segmentation, object detector, or 3D CNN.
- **Data:** public CT/X-ray/ultrasound data with masks or boxes.
- **Difficulty:** advanced.
- **GPU:** strongly recommended.
- **Output:** regions, masks, API/demo, experiment registry, and more formal
  validation protocol.

### Selected option

Choose Option A first. A smaller complete system is better proof of work than
an unfinished “advanced” system. Once the baseline is documented, add Option B
as an explicitly labeled experiment rather than replacing the baseline.

## C. Selected project contract

| Item | Decision |
|---|---|
| Task | Binary classification: normal vs pneumonia |
| Dataset | Official MedMNIST PneumoniaMNIST |
| Input | Grayscale, resized to 64×64 |
| Target | `0 = normal`, `1 = pneumonia` |
| Model | Compact CNN |
| Selection metric | Validation AUC |
| Evaluation | Official test split, accuracy/AUC/precision/recall/confusion matrix |
| Reproducibility | Seed 42, saved history, saved model, explicit CLI |
| Privacy | No real patient data, no credentials, no hospital integration |
| Safety | Educational use only |

## D. Architecture and data flow

```text
official public data
        ↓
split-aware loader
        ↓
resize + grayscale + [0,1] normalization
        ↓
CNN features: Conv2D → BatchNorm → Pooling (repeated)
        ↓
GAP → Dense → Dropout → sigmoid
        ↓
pneumonia probability
        ↓
thresholded educational label + metrics + plots
```

### Module responsibilities

- `config.py`: paths and defaults.
- `labels.py`: safe label mapping.
- `data.py`: official dataset access and deterministic subsets.
- `preprocessing.py`: image shape/range contract.
- `model.py`: compiled baseline.
- `training.py`: callbacks, class weighting, checkpointing, history.
- `evaluation.py`: test metrics and artifact creation.
- `prediction.py`: one-image inference.
- `visualization.py`: plots and prediction cards.
- `main.py`: command-line user experience.

## E. Folder structure and file purpose

```text
AI-Powered-Medical-Image-Analysis/
├── data/raw/          downloaded `.npz` data; ignored
├── data/processed/    optional future derived data; ignored
├── docs/              specification and learning guide
├── images/            reviewed portfolio screenshots only
├── models/            trained model files; ignored
├── notebooks/         optional exploratory notebooks
├── outputs/           local metrics and plots; ignored
├── src/               reusable Python modules
├── tests/             unit tests
├── main.py            CLI entry point
├── README.md          public project documentation
├── requirements.txt   reproducible dependency ranges
└── .gitignore         prevents data, models, secrets, and caches in Git
```

The `.gitkeep` files keep the intended empty directories visible in a fresh
clone. The folders are empty until you run the workflow locally.

## F. Installation and setup

### Windows

```powershell
cd "F:\Projects AI\AI-Powered Medical Image Analysis"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### macOS/Linux

```bash
cd "/path/to/AI-Powered Medical Image Analysis"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Use Python 3.11 or 3.12 if TensorFlow installation does not support the
interpreter already on your machine. The exact Python and TensorFlow versions
must be written in your final experiment notes.

### First verification

```bash
python --version
python -m pip show tensorflow numpy opencv-python-headless
python -m unittest discover -s tests -v
```

Do not continue to a full training run if this basic test fails. Fix the
environment first.

## G. Implementation walkthrough

### 1. Load data

`src/data.py` downloads the official MedMNIST NumPy archive from Zenodo,
verifies its MD5 checksum, selects an explicit `train`, `val`, or `test` split,
and stores the file under `data/raw/`. The function returns raw arrays to the
preprocessing step. The optional `max_samples` flag creates a small
deterministic, approximately class-balanced subset for smoke tests.

Important rule: never use `max_samples` as a substitute for a final experiment.
It exists to prove the pipeline wiring before spending time on a full run.

### 2. Preprocess images

`src/preprocessing.py` enforces this contract:

```text
input:  (N, H, W) uint8 grayscale
output: (N, 64, 64, 1) float32 in [0, 1]
```

OpenCV uses area interpolation for downsampling. The fallback Pillow path is
there so the pure preprocessing tests remain understandable, but the normal
environment installs OpenCV.

### 3. Build the CNN

The compact model is intentionally readable:

```text
Input(64,64,1)
RandomRotation + small RandomZoom
Conv2D(32) + BatchNormalization + MaxPooling
Conv2D(64) + BatchNormalization + MaxPooling
Conv2D(128) + BatchNormalization + MaxPooling
GlobalAveragePooling2D
Dropout + Dense(64) + Dropout
Dense(1, sigmoid)
```

Binary cross-entropy is appropriate for a single sigmoid output. AUC,
accuracy, precision, and recall are tracked because healthcare-oriented
screening discussions should not rely on accuracy alone.

### 4. Train responsibly

`src/training.py`:

- calculates class weights from training labels only;
- uses `ModelCheckpoint` to save the best validation-AUC model;
- uses `EarlyStopping` to restore the best weights;
- reduces the learning rate when validation loss stalls; and
- saves JSON history for later inspection.

Do not tune hyperparameters by repeatedly looking at the test set. Use train and
validation data for development, then use test data for the final report.

### 5. Evaluate

`src/evaluation.py` computes:

- Keras loss, accuracy, AUC, precision, and recall;
- ROC AUC from probability scores;
- a labeled 2×2 confusion matrix; and
- a classification report with per-class precision, recall, and F1.

The output is written to `outputs/metrics.json`. Copy actual values into the
README only after the run completes. Include the run configuration alongside
the metrics so a reader knows what was measured.

### 6. Predict one local image

`src/prediction.py` applies the same preprocessing used during training, which
is essential. If a different resize or normalization rule is used at inference,
the model is no longer seeing the type of input it learned from.

The response includes both the probability and the thresholded label. It also
includes `clinical_use: false` so a demo screenshot cannot be mistaken for a
clinical result.

## H. Virtual simulation procedure

1. Choose a public image dataset with a license that permits your educational
   use.
2. Download it through the official Zenodo release, not a random mirror.
3. Store it in `data/raw/`, which is Git-ignored.
4. Inspect class examples with `sample_grid.png`.
5. Train the baseline with a fixed seed.
6. Save the checkpoint and history.
7. Evaluate on the test split.
8. Review the confusion matrix and false predictions.
9. Run one local image through the same preprocessing path.
10. Capture the command, output artifacts, environment, and limitations.

### What the simulation represents

| Prototype part | Real-world analogue | Missing in this project |
|---|---|---|
| MedMNIST image | Imaging archive | Clinical acquisition variability and metadata |
| CNN probability | Screening assistant | Calibration and clinical validation |
| Test metrics | Offline benchmark | External-site performance |
| PNG prediction card | Review UI mock | Secure PACS/RIS integration and audit trail |

The table makes the simulation realistic without pretending that it is a real
hospital deployment.

## I. Run the project

### Smoke run first

Use this before a full dataset run:

```bash
python main.py --mode train --max-samples 64 --epochs 1 --batch-size 16
```

This should download or locate the selected data, create a sample grid, train
one epoch, save a model, and create evaluation artifacts. It is a wiring check,
not a meaningful final benchmark.

### Full baseline

```bash
python main.py --mode train --epochs 20 --batch-size 32
```

The first run may take time because the dataset is downloaded and CPU training
is slower than GPU training. Do not interrupt the first run just because the
terminal is quiet for a moment; inspect resource usage and logs.

### Evaluate a saved model

```bash
python main.py --mode evaluate --no-download
```

The model must exist at `models/pneumonia_cnn.keras`. Evaluation uses the test
split and writes fresh `metrics.json` and figures.

### Predict a local image

```bash
python main.py --mode predict --image "path/to/xray.png"
```

The image should be a grayscale or readable chest X-ray-style file for a demo.
The program rejects a missing/unreadable file. It does not verify that the
image is actually a chest X-ray, so the human operator must control the input.

### Outputs

```text
models/pneumonia_cnn.keras
outputs/history.json
outputs/metrics.json
outputs/sample_grid.png
outputs/training_curves.png
outputs/confusion_matrix.png
outputs/prediction_examples.png
outputs/single_prediction.png
```

The exact metric values are intentionally not documented in advance. A metric
is meaningful only with its run configuration, dataset version, split, and
actual artifact.

## J. GitHub upload strategy

### Before Git

Check the files that will be committed:

```bash
git init
git branch -M main
git status --short
git status --ignored
git add .
git diff --cached
```

Look for `.env`, credentials, data files, model binaries, and patient images.
If any appear staged, stop and fix `.gitignore` before committing.

### Recommended repository

- Name: `ai-powered-medical-image-analysis`
- Description: `Educational public-data chest X-ray classification pipeline with TensorFlow/Keras and reproducible visual evaluation.`
- Topics: `medical-imaging`, `computer-vision`, `deep-learning`, `tensorflow`,
  `keras`, `python`, `medmnist`, `machine-learning`, `portfolio-project`.

### Initial push

```bash
git commit -m "chore: initialize medical image analysis project"
git remote add origin https://github.com/YOUR_USERNAME/ai-powered-medical-image-analysis.git
git push -u origin main
```

Replace the placeholder with the actual repository URL. Use GitHub-supported
authentication; never commit a token or paste one into a source file.

### Large assets

Do not commit the raw dataset. For selected screenshots, use `images/`. For a
large demo recording, use a GitHub release, Git LFS, or a portfolio link. Every
asset needs a short caption saying what was run and what the asset proves.

## K. README and professional documentation

A strong README answers these questions in order:

1. What does this system do?
2. What is educational and what is not clinical?
3. Which dataset and license are used?
4. How is the architecture structured?
5. How can a new user install and run it?
6. What are the actual measured results?
7. What errors and limitations were found?
8. What future work is realistic?

The repository README contains this content. After your first full run, add a
results row copied from `metrics.json`, plus 3–5 reviewed images. Do not leave
placeholder metrics in public documentation.

### Suggested results table

```markdown
| Run ID | Image size | Epochs | Test accuracy | Test AUC | Recall | Hardware |
|---|---:|---:|---:|---:|---:|---|
| 2026-10-13-seed42 | 64 | 20 | paste actual | paste actual | paste actual | CPU/GPU |
```

The date is only an example. Use the actual local date and run configuration.

## L. Seven-day proof plan

### Day 1 — setup

- Install the environment.
- Run tests.
- Commit: `chore: initialize reproducible Python project`.
- Proof: terminal showing interpreter, install, and tests.

### Day 2 — dataset

- Read the official dataset page.
- Record labels, license, source, and split sizes.
- Commit: `docs: document MedMNIST provenance and data policy`.
- Proof: source link, dataset tree, no data staged.

### Day 3 — preprocessing

- Run sample grid and preprocessing tests.
- Explain resize, normalization, and channel dimension.
- Commit: `feat: add deterministic image preprocessing`.
- Proof: sample image and shape/range test.

### Day 4 — model

- Read `src/model.py` and explain each layer.
- Run smoke training.
- Commit: `feat: add compact CNN baseline`.
- Proof: model summary and architecture diagram.

### Day 5 — training

- Run a documented baseline.
- Save history and best checkpoint.
- Commit: `feat: add checkpointed model training`.
- Proof: training log, history JSON, curves.

### Day 6 — evaluation

- Run frozen test evaluation.
- Inspect false positives and false negatives.
- Commit: `feat: add holdout evaluation and visual evidence`.
- Proof: metrics JSON and confusion matrix.

### Day 7 — publishing

- Clean the repository.
- Add screenshots and citations.
- Commit: `docs: publish reproducible GitHub demo guide`.
- Proof: public clone, README run instructions, optional demo video.

## M. Screenshot and demo asset checklist

### Required assets

1. `01-sample-grid.png` — labeled dataset examples.
2. `02-preprocessing.png` — resized/normalized input explanation.
3. `03-training-log.png` — terminal training output.
4. `04-training-curves.png` — train/validation loss and metrics.
5. `05-confusion-matrix.png` — class-specific errors.
6. `06-predictions.png` — correct and incorrect example predictions.
7. `07-single-prediction.png` — educational one-image output.
8. `08-github-repo.png` — repository page showing README and structure.

### Capture rules

- Do not show a real patient's name, ID, date of birth, or DICOM metadata.
- Do not show secrets, local tokens, or private file paths in a public capture.
- Include the command or run ID where possible.
- Keep the safety disclaimer visible in the README and demo.
- Use filenames that sort in presentation order.
- Store working artifacts under `outputs/`; copy reviewed assets to `images/`.

### 60–90 second demo script

1. “This is an educational public-data project, not a clinical tool.”
2. “It uses official PneumoniaMNIST data with normal and pneumonia labels.”
3. “The pipeline preprocesses images and trains a compact CNN.”
4. “Validation AUC selects the saved model; the test split is held out.”
5. “The confusion matrix makes errors visible instead of hiding them.”
6. “This single-image result is a model estimate for the benchmark task only.”
7. “Future work would require external validation, calibration, privacy,
   monitoring, and clinical governance.”

## Proof checklist

### Reproducibility

- [ ] Python version recorded.
- [ ] Dependency install works from a clean environment.
- [ ] Official dataset link and license cited.
- [ ] Dataset is not committed.
- [ ] Seed and run flags recorded.
- [ ] Train/validation/test roles are explained.

### Engineering

- [ ] Code is modular.
- [ ] CLI commands are copy-paste ready.
- [ ] Tests cover labels and preprocessing.
- [ ] Checkpoint and history are saved.
- [ ] Evaluation produces machine-readable JSON.
- [ ] Visual artifacts expose errors.
- [ ] `.gitignore` protects data, models, secrets, and caches.

### Communication

- [ ] README starts with a safety disclaimer.
- [ ] No invented results appear.
- [ ] Metrics include more than accuracy.
- [ ] Limitations are specific.
- [ ] Screenshots have captions.
- [ ] Git history has small, meaningful commits.
- [ ] Future extensions are labeled as future work.

## Sources

- MedMNIST: <https://medmnist.com/>
- MedMNIST source repository: <https://github.com/MedMNIST/MedMNIST>
- Official data record: <https://doi.org/10.5281/zenodo.10519652>
- Scientific Data paper: <https://doi.org/10.1038/s41597-022-01721-8>
- Keras callbacks: <https://keras.io/api/callbacks/model_checkpoint/> and
  <https://keras.io/api/callbacks/early_stopping/>
- TensorFlow install: <https://www.tensorflow.org/install/pip>
