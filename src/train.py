"""Train and fine-tune a binary MobileNetV2 leaf classifier."""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "plant_disease_mobilenetv2.keras"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "results" / "training_history.json"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_LAYERS = 30
RANDOM_SEED = 42


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_datasets(data_dir: Path) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    common = dict(image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="binary")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train", shuffle=True, seed=RANDOM_SEED, **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val", shuffle=False, **common
    )
    if train_ds.class_names != val_ds.class_names:
        raise ValueError("Training and validation class names do not match.")
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    return (
        train_ds.cache().prefetch(autotune),
        val_ds.cache().prefetch(autotune),
        class_names,
    )


def build_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build the architecture used by the supplied train.py."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,), name="image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="disease_probability")(x)
    return tf.keras.Model(inputs, outputs), base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )


def merge_histories(*histories: tf.keras.callbacks.History) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


def save_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    """Save loss and accuracy curves from both training phases."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Binary cross-entropy")
    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for axis in axes:
        axis.axvline(HEAD_EPOCHS - 0.5, color="gray", linestyle="--", label="fine-tuning")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--history-path", type=Path, default=DEFAULT_HISTORY_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seed(RANDOM_SEED)
    train_ds, val_ds, class_names = load_datasets(args.data_dir)
    if class_names != ["diseased", "healthy"]:
        raise ValueError(f"Expected classes ['diseased', 'healthy'], found {class_names}")

    model, base_model = build_model()
    compile_model(model, HEAD_LEARNING_RATE)
    print("Phase 1: training the classification head")
    head_history = model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS)

    print("Phase 2: fine-tuning the last MobileNetV2 layers")
    base_model.trainable = True
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False
    compile_model(model, FINE_TUNE_LEARNING_RATE)
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=HEAD_EPOCHS,
        epochs=HEAD_EPOCHS + FINE_TUNE_EPOCHS,
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    history = merge_histories(head_history, fine_tune_history)
    args.history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_training_curves(history, args.history_path.parent / "training_curves.png")
    print(f"Model saved to: {args.model_path.resolve()}")


if __name__ == "__main__":
    main()
