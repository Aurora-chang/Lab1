"""Predict one image, or evaluate an entire labeled test split."""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader

from dataset import ChestXrayDataset, make_transform, scan_split, CLASSES
from model import build_model, choose_device
from train import run_epoch
from plots import plot_confusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path)
    group.add_argument("--test-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("evaluation"))
    args = parser.parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint["classes"] != CLASSES:
        raise ValueError("Checkpoint has incompatible class ordering")
    model = build_model(checkpoint["model"], pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = make_transform(checkpoint["size"])
    if args.image:
        with Image.open(args.image) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = model(tensor).softmax(dim=1)[0].cpu().tolist()
        print(json.dumps({"image": str(args.image), "prediction": CLASSES[probabilities.index(max(probabilities))],
                          "probabilities": dict(zip(CLASSES, probabilities))}, indent=2))
    else:
        dataset = ChestXrayDataset(scan_split(args.test_dir), transform)
        metrics = run_epoch(model, DataLoader(dataset, batch_size=args.batch_size), device)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
        plot_confusion(metrics["confusion_matrix"], args.output / "confusion_matrix.png")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
