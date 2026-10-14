"""Download the assignment dataset from Kaggle and locate its split root."""
import argparse
import json
from pathlib import Path

DATASET_HANDLE = "paultimothymooney/chest-xray-pneumonia"
CLASSES = ("NORMAL", "PNEUMONIA")


def find_dataset_root(download_directory):
    directory = Path(download_directory)
    candidates = {directory}
    candidates.update(path.parent for path in directory.rglob("train") if path.is_dir())
    valid = [path for path in candidates if all(
        (path / split / name).is_dir() for split in ("train", "test") for name in CLASSES)]
    if not valid:
        raise FileNotFoundError(f"Could not find train/test with NORMAL/PNEUMONIA under {directory}")
    # Some Kaggle archive versions include duplicate nested copies; use one root.
    return sorted(valid, key=lambda path: (len(path.parts), str(path)))[0]


def main():
    import kagglehub
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/dataset_location.json"))
    args = parser.parse_args()
    downloaded = Path(kagglehub.dataset_download(DATASET_HANDLE))
    root = find_dataset_root(downloaded)
    metadata = {"handle": DATASET_HANDLE, "download_directory": str(downloaded), "dataset_root": str(root)}
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
