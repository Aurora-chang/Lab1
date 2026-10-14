"""Custom dataset and loaders. NORMAL=0, PNEUMONIA=1."""
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASSES = ["NORMAL", "PNEUMONIA"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_split(root):
    root = Path(root)
    samples = []
    for label, name in enumerate(CLASSES):
        folder = root / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected class folder: {folder}")
        paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS)
        if not paths:
            raise ValueError(f"No images found in {folder}")
        samples.extend((str(path), label) for path in paths)
    return samples


def make_transform(size=224, augment=False):
    steps = [transforms.Resize((size, size))]
    if augment:
        steps += [transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.1, contrast=0.1)]
    steps += [transforms.ToTensor(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(steps)


class ChestXrayDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return image, label


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_loaders(root, batch_size, size, workers, seed, val_fraction=0.2,
                  use_official_val=False, augment=True):
    root = Path(root)
    train_samples = scan_split(root / "train")
    if use_official_val:
        val_samples = scan_split(root / "val")
    else:
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")
        # Stratify using training data only; test images never enter validation.
        rng = random.Random(seed)
        train_parts, val_samples = [], []
        for label in range(2):
            group = [s for s in train_samples if s[1] == label]
            if len(group) < 2:
                raise ValueError("Each training class needs at least two images")
            rng.shuffle(group)
            count = min(len(group) - 1, max(1, round(len(group) * val_fraction)))
            val_samples.extend(group[:count])
            train_parts.extend(group[count:])
        train_samples = train_parts
    test_samples = scan_split(root / "test")
    samples = {"train": train_samples, "val": val_samples, "test": test_samples}
    loaders = {}
    for split, items in samples.items():
        dataset = ChestXrayDataset(items, make_transform(size, augment and split == "train"))
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=split == "train", num_workers=workers,
            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))
    return loaders, samples
