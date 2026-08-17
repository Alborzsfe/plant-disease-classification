"""Merge PlantVillage classes into binary healthy/diseased splits."""

import argparse
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "PlantVillage"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def copy_split(raw_dir: Path, output_dir: Path, split_name: str) -> tuple[int, int]:
    """Copy one split while merging source folders into two target classes."""
    split_dir = raw_dir / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    counts = {"healthy": 0, "diseased": 0}
    for class_folder in sorted(split_dir.iterdir()):
        if not class_folder.is_dir():
            continue

        target_class = "healthy" if "healthy" in class_folder.name.lower() else "diseased"
        target_dir = output_dir / split_name / target_class
        target_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(class_folder.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                destination = target_dir / f"{class_folder.name}__{image_path.name}"
                shutil.copy2(image_path, destination)
                counts[target_class] += 1

    return counts["healthy"], counts["diseased"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split_name in ("train", "val"):
        healthy, diseased = copy_split(args.raw_dir, args.output_dir, split_name)
        print(f"{split_name}: healthy={healthy}, diseased={diseased}")
    print(f"Prepared data written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
