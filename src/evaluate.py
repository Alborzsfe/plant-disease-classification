"""Evaluate a saved binary classifier on the validation split."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "val"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "plant_disease_mobilenetv2.keras"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def save_confusion_matrix(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(2), yticks=range(2), xticklabels=labels, yticklabels=labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Validation Confusion Matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    dataset = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )
    if dataset.class_names != ["diseased", "healthy"]:
        raise ValueError(f"Unexpected class order: {dataset.class_names}")

    model = tf.keras.models.load_model(args.model_path, compile=False)
    probabilities = model.predict(dataset, verbose=1).reshape(-1)
    true_labels = np.concatenate([labels.numpy().reshape(-1) for _, labels in dataset]).astype(int)
    predicted_labels = (probabilities >= THRESHOLD).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="binary", pos_label=1, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision_healthy": float(precision),
        "recall_healthy": float(recall),
        "f1_healthy": float(f1),
        "decision_threshold": THRESHOLD,
        "positive_class": "healthy",
        "samples": int(len(true_labels)),
    }
    matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(matrix, dataset.class_names, args.results_dir / "confusion_matrix.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
