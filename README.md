# Binary Plant Leaf Disease Classification with MobileNetV2

## Overview

This university portfolio project performs binary classification of leaf images
as `healthy` or `diseased`. It uses TensorFlow/Keras, an ImageNet-pretrained
MobileNetV2 backbone, and transfer learning.

## Problem

The original PlantVillage category folders are merged into two classes. A folder
whose name contains `healthy` is mapped to `healthy`; every other folder is mapped
to `diseased`. This repository documents that exact rule rather than claiming
individual disease identification.

## Approach

The pipeline prepares the binary directory structure, validates image counts,
trains a frozen MobileNetV2 classification head, fine-tunes the last 30 backbone
layers, evaluates the validation split, and supports inference on one image.

## Dataset

The supplied code identifies the dataset as PlantVillage. The dataset is not
included in this repository. The exact source URL, release, license, image count,
and split procedure were not recorded in the supplied files and must be verified
before publication. See [`data/README.md`](data/README.md) for the expected layout.

## Model Architecture

- Input: RGB image, `224 × 224 × 3`
- Preprocessing: `tf.keras.applications.mobilenet_v2.preprocess_input`
- Backbone: MobileNetV2 with ImageNet weights and `include_top=False`
- Pooling: Global Average Pooling
- Regularization: Dropout with rate `0.2`
- Output: one sigmoid unit

With alphabetical directory labels, `diseased=0` and `healthy=1`; the sigmoid
value is therefore interpreted as the probability of `healthy`.

## Training

Phase 1 freezes MobileNetV2 and trains the classification head for 10 epochs using
Adam (`learning_rate=1e-3`) and binary cross-entropy. Images are loaded in batches
of 32. The current repository intentionally preserves the supplied script's lack
of data augmentation.

## Fine-tuning

Phase 2 makes the backbone trainable, keeps all but its last 30 layers frozen,
and continues for 5 epochs using Adam (`learning_rate=1e-5`). The backbone is
called with `training=False`, preserving inference behavior for layers such as
Batch Normalization during the forward pass.

## Evaluation

`evaluate.py` calculates accuracy plus binary precision, recall, and F1 for the
positive class `healthy` at a threshold of 0.5. It saves `results/metrics.json`
and `results/confusion_matrix.png`. These are validation metrics, not independent
test-set metrics.

## Results

- Validation accuracy: `[Add validation accuracy here]`
- Healthy-class precision: `[Add validation precision here]`
- Healthy-class recall: `[Add validation recall here]`
- Healthy-class F1-score: `[Add validation F1-score here]`

No training logs or trustworthy metric values were supplied, so no performance
number is claimed. Run evaluation and replace the placeholders with generated
values, preferably after adding a separate held-out test set.

## Inference

```bash
python src/predict.py path/to/leaf.jpg
```

The command prints the predicted class and its model-derived confidence. It is a
research demonstration and must not be treated as professional agronomic advice.

## Project Structure

```text
plant-disease-classification/
├── README.md
├── PROJECT_AUDIT.md
├── LICENSE
├── requirements.txt
├── data/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
│   └── README.md
├── results/
│   └── .gitkeep
└── src/
    ├── prepare_data.py
    ├── check_data.py
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

## Installation

Python 3 and a virtual environment are recommended. Exact package versions were
not present in the original project, so they are deliberately not invented here.

```bash
git clone <your-repository-url>
cd plant-disease-classification
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For exact reproducibility after confirming a working environment, record versions
with `python -m pip freeze` or adopt a tested version range.

## Usage

```bash
# 1. Put the source data under data/raw/PlantVillage/train and val
python src/prepare_data.py

# 2. Check the processed folders and counts
python src/check_data.py

# 3. Train and fine-tune
python src/train.py

# 4. Evaluate the saved model
python src/evaluate.py

# 5. Predict one image
python src/predict.py path/to/leaf.jpg
```

Each script also supports `--help` and optional path overrides.

## Technologies

- Python
- TensorFlow/Keras
- MobileNetV2 transfer learning
- NumPy
- scikit-learn
- Matplotlib

## Future Improvements

- document the exact PlantVillage source, release, license, and split;
- add a leakage-checked held-out test set;
- record dataset counts and class balance;
- add verified augmentation and compare it experimentally;
- add callbacks and save the best validation checkpoint;
- export reproducible environment versions;
- investigate calibration and performance across plant species and diseases;
- test on real-world images outside PlantVillage's controlled backgrounds.

## License

No license has been selected for this project. Add a license only after choosing
appropriate terms and separately comply with the dataset's license.

For a detailed inspection of the original scripts and supplied H5 artifact, see
[`PROJECT_AUDIT.md`](PROJECT_AUDIT.md).
