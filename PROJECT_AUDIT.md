# Audit of the supplied project

## What the original pipeline does

1. `prepare_data.py` reads already-separated `train` and `val` folders from a
   local PlantVillage directory.
2. It merges all original folders containing `healthy` into a binary `healthy`
   folder and every other folder into `diseased`.
3. `check_data.py` checks the four expected folders and counts JPG, JPEG, and PNG
   files directly inside them.
4. `train.py` loads the processed folders at 224 × 224 pixels with binary labels.
5. It builds an ImageNet-pretrained MobileNetV2 without its original classifier,
   applies MobileNetV2 preprocessing, global average pooling, dropout (0.2), and
   a one-unit sigmoid output.
6. Phase 1 freezes MobileNetV2 and trains the new head for 10 epochs with Adam
   (`1e-3`). Phase 2 unfreezes the backbone, freezes all but its last 30 layers,
   and trains for 5 more epochs with Adam (`1e-5`).
7. The original script records only accuracy and saves an HDF5 whole model.

## Inputs, labels, and outputs

- Model input: RGB image tensor shaped `(224, 224, 3)`.
- Directory class order: alphabetical, therefore `diseased=0`, `healthy=1` for
  the expected folder names.
- Model output: one sigmoid value interpreted as the probability of class 1,
  which is `healthy` under that ordering.
- Original preprocessing: image resizing by the dataset loader followed by
  `mobilenet_v2.preprocess_input` inside the model (approximately maps pixels
  from `[0, 255]` to `[-1, 1]`).
- Original data augmentation: none in the supplied `train.py`.

## Supplied H5 inspection

The artifact is a 21 MiB HDF5 file and contains a serialized Keras Functional
model. Its configuration shows:

- input `(None, 224, 224, 3)`;
- internal horizontal flip, rotation (factor ±0.05), and zoom (factor 0.1);
- MobileNetV2 (`mobilenetv2_1.00_224`);
- MobileNetV2-style division by 127.5 and subtraction of 1;
- global average pooling, dropout 0.2, and one sigmoid Dense output.

It is therefore **not an exact match** for the supplied `train.py`, because the
artifact includes augmentation layers that the script does not build. The
overall backbone and classification head are otherwise consistent. Its config
also contains serialized `TrueDivide` and `Subtract` operation layers. These can
cause `Unknown layer: 'TrueDivide'` when an H5 model produced by one Keras stack
is loaded by an incompatible TensorFlow/Keras version.

The file's HDF5 container and serialized configuration are readable. A complete
runtime load and numerical inference test could not be performed in the audit
environment because TensorFlow and h5py were not installed. Consequently this
report does not claim that the artifact is fully loadable in every environment.

## Gaps and scientific limitations

- No exact dataset URL, release, license, image counts, or split-generation
  method is documented.
- The source data must already contain train/validation splits; no test set is
  used. Reporting validation performance as final test performance would be
  scientifically weak.
- Grouping every non-healthy class as diseased assumes all other folders truly
  represent disease. Folder names should be audited.
- Re-running preparation over changed data does not clear stale output files.
- Filename prefixing reduces collisions but does not detect duplicates or
  leakage across train and validation sets.
- The original training has no explicit random seed, callbacks, class weighting,
  checkpoint selection, training-curve export, or non-accuracy metrics.
- In-memory `.cache()` may be unsuitable for a dataset larger than available RAM.
- Fine-tuning includes trainable Batch Normalization layers, although the backbone
  is invoked with `training=False`. This choice should be tested and documented.
- No training logs were supplied. Accuracy, precision, recall, F1, dataset size,
  and performance cannot be truthfully reported yet.
- The H5 artifact cannot be guaranteed reproducible from the supplied training
  script because its augmentation pipeline is absent from that script.

## Changes made for the repository

Absolute Windows paths were replaced by project-relative defaults with optional
CLI overrides. Scripts were organized into functions, guarded with `main`, and
given validation and error messages. A fixed seed was added for improved (not
perfect) reproducibility. The original network architecture was retained and no
augmentation was added. New training runs use `.keras`; evaluation and inference
are provided for models produced by the repository's current `train.py`.
