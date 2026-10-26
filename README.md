# Chrysanthemum Recognition System

An individual undergraduate dissertation project by **Yuhan Wei**, Nanjing Agricultural University, 2026. The project connects an eight-class flower dataset, an EfficientNet-B3 classifier, and a Flask application for image upload, preview, and top-three predictions with confidence scores.

## Project summary

- Dataset described in the dissertation: **1,169 images across eight classes**, divided into training, validation, and test sets.
- Backbone: ImageNet-pretrained EfficientNet-B3, with the final five feature modules unfrozen and a custom classifier.
- Training: random resized cropping, horizontal flips, rotation, colour jitter, AdamW, cosine learning-rate scheduling, and early stopping.
- Best recorded validation accuracy: **95.38%**. This is a historical checkpoint result, not a benchmark rerun for this release. Different data splits or training runs can produce different results.
- Web interface: image selection, preview, and top-three class predictions with confidence scores.

The final implementation published here uses EfficientNet-B3 without an additional CBAM module and uses unweighted cross-entropy loss. The original report discusses additional model variants; those variants are not implemented in this release. No unverified test accuracy or ablation results are claimed here.

## Files

| File | Purpose |
| --- | --- |
| `train_top8.py` | Train, select the best validation checkpoint, and evaluate on the test set |
| `deploy_top8.py` | Run the local Flask application and prediction API |
| `templates/index_top8.html` | English user interface with inline CSS and JavaScript |
| `split_dataset.py` | Copy raw images into train/validation/test folders |
| `check_images.py` | Report unreadable images without deleting files |
| `confusion_top8.py` | Generate a confusion matrix and per-class classification report |
| `requirements.txt` | Runtime and evaluation dependencies |

## Setup

Use **Python 3.11** and run the commands from the repository root. The dependency ranges select compatible PyTorch 2.5 / torchvision 0.20 release families. This publication copy has not been checked through a full dependency installation or a new training run.

```bash
python -m venv .venv
```

Activate the environment on Linux/macOS:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the application

Place a trusted, compatible **`best_top8.pth`** checkpoint in the repository root, then run:

```bash
python deploy_top8.py
```

Open **http://127.0.0.1:5000**. The checkpoint and original images are not included in this repository. Without the checkpoint, model inference cannot start. You can generate a checkpoint by training on a correctly structured dataset; original images may require separate permission for redistribution. The HTML file must be served through Flask for prediction to work; opening it directly or hosting it alone on GitHub Pages does not run the Python backend.

`MODEL_PATH` can select a different checkpoint path. The backend uses the checkpoint's class names when provided and returns its recorded validation accuracy as metadata, without substituting a hardcoded test accuracy.

## Data and training

The eight source folder labels are:

`乒乓菊`, `勋章菊`, `墨菊`, `波斯菊`, `金丝皇菊`, `金鸡菊`, `雏菊`, `黑眼菊`.

These are the project's dataset labels; the collection includes flowers from different botanical genera. The English interface uses transliterations rather than claiming that all labels represent taxonomic Chrysanthemum cultivars.

Place the reviewed source images in `chrysanthemum_dataset/<class-name>/`, then run:

```bash
python split_dataset.py
python check_images.py
python train_top8.py
python confusion_top8.py
```

The split script uses seed 42 and approximately 70% / 15% / 15% per class, with integer rounding. It refuses to write into a nonempty destination so that repeated splitting does not leave overlapping copies. It does not detect visually duplicated source images: inspect and remove duplicates before splitting. Sorted input filenames make reruns deterministic for the same source folder contents. Rebuilding the split is not guaranteed to reconstruct the original experiment's split or metrics.

The output structure must contain `data/train/<class-name>/`, `data/val/<class-name>/`, and `data/test/<class-name>/` for all eight classes. Set `RAW_DATA_DIR` and `DATA_DIR` environment variables to use alternative locations. Training downloads ImageNet weights on first use and writes `best_top8.pth` in the current working directory. Evaluation also reads that checkpoint from the current working directory.

## API

- `GET /`: English application page.
- `GET /api/info`: model description, class names, and recorded validation accuracy when available.
- `POST /api/predict`: multipart field `image`, or JSON field `image` containing a Base64 image or data URL. Returns `predictions` and `top1`; each prediction contains `class` and `confidence` (0 to 1).

Confidence scores are model outputs, not a guarantee of correct classification. The system has no explicit unknown-class rejection. Additional external evaluation is needed before use in unfamiliar acquisition conditions.

## Publication changes

This copy keeps the supplied classifier and training hyperparameters. Personal machine paths were replaced by configurable paths; the web interface was translated into English; hardcoded test accuracy was removed; checkpoint labels are checked during evaluation; and invalid input handling was improved. Dataset splitting now sorts filenames and prevents output reuse. Image checking reports problems without automatic deletion. The API defaults to local access.

The original images, model weights, and historical experiment logs are not bundled. No new model training or checkpoint inference was run while preparing this publication copy.
