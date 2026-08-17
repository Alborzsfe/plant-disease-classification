"""Validate the expected binary dataset layout and count image files."""

import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def count_images(directory: Path) -> int:
    """Count supported image files directly inside a directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Required directory not found: {directory}")
    return sum(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in directory.iterdir()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failed = False
    for split_name in ("train", "val"):
        for class_name in ("healthy", "diseased"):
            directory = args.data_dir / split_name / class_name
            try:
                count = count_images(directory)
                print(f"{split_name}/{class_name}: {count} images ({directory})")
                failed = failed or count == 0
            except FileNotFoundError as error:
                print(f"ERROR: {error}")
                failed = True
    if failed:
        raise SystemExit("Dataset check failed: a required folder is missing or empty.")


if __name__ == "__main__":
    main()
