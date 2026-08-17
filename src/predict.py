"""Run inference for one leaf image."""

import argparse
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "plant_disease_mobilenetv2.keras"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ("diseased", "healthy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a JPG, JPEG, or PNG image")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    model = tf.keras.models.load_model(args.model_path, compile=False)
    image = tf.keras.utils.load_img(args.image, target_size=IMAGE_SIZE, color_mode="rgb")
    batch = tf.expand_dims(tf.keras.utils.img_to_array(image), axis=0)
    healthy_probability = float(model.predict(batch, verbose=0)[0][0])
    predicted_index = int(healthy_probability >= 0.5)
    confidence = healthy_probability if predicted_index == 1 else 1.0 - healthy_probability
    print(f"Prediction: {CLASS_NAMES[predicted_index]}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
