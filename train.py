"""Train one ResNet, select on validation F1, then evaluate test once."""
import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torchvision
from tqdm import tqdm

from dataset import build_loaders, CLASSES
from model import build_model, choose_device
from metrics import classification_metrics
from plots import plot_history, plot_confusion


def run_epoch(model, loader, device, criterion=None, optimizer=None):
    training = optimizer is not None
    model.train(training)
    matrix = torch.zeros(2, 2, dtype=torch.int64)
    loss_sum, batches = 0.0, 0
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc="train" if training else "evaluate", leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if criterion is not None:
                # Sum reduction / image count gives exact dataset-level weighted loss.
                summed_loss = criterion(logits, labels)
                loss_sum += summed_loss.item()
                if training:
                    (summed_loss / labels.numel()).backward()
                    optimizer.step()
            encoded = (2 * labels + logits.argmax(dim=1)).detach().cpu()
            matrix += torch.bincount(encoded, minlength=4).reshape(2, 2)
            batches += 1
    if not batches:
        raise ValueError("Empty data loader")
    result = classification_metrics(matrix.tolist())
    if criterion is not None:
        result["loss"] = loss_sum / result["samples"]
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--official-val", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.size < 32 or args.workers < 0:
        parser.error("epochs >= 1, batch-size >= 2, size >= 32, workers >= 0 required")
    return args


def main():
    args = parse_args()
    output = args.output / args.model
    # Keep runs separate to prevent accidentally mixing curves and checkpoints.
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; use a new --output directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = choose_device(args.device)
    loaders, samples = build_loaders(
        args.dataset, args.batch_size, args.size, args.workers, args.seed,
        args.val_fraction, args.official_val, not args.no_augmentation)
    # Avoid a singleton final training batch (ResNet BatchNorm at small input sizes).
    if len(samples["train"]) % args.batch_size == 1:
        raise ValueError("Training leaves a singleton batch; choose another batch size")
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device=str(device), torch_version=str(torch.__version__),
                  torchvision_version=str(torchvision.__version__))
    (output / "config.json").write_text(json.dumps(config, indent=2))
    manifest = {split: [{"path": str(Path(path).relative_to(args.dataset)), "label": label}
                        for path, label in items] for split, items in samples.items()}
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    counts = np.bincount([label for _, label in samples["train"]], minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="sum")
    model = build_model(args.model, not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history, best_f1, best_epoch = [], -1.0, 0
    start = time.perf_counter()
    print(f"Model={args.model}, device={device}, class counts={counts.tolist()}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, loaders["train"], device, criterion, optimizer)
        val = run_epoch(model, loaders["val"], device, criterion)
        row = {"epoch": epoch}
        for split, metrics in (("train", train), ("val", val)):
            row.update({f"{split}_{key}": metrics[key] for key in ("loss", "accuracy", "f1")})
        history.append(row)
        with (output / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"Epoch {epoch}/{args.epochs} train acc={train['accuracy']:.4f} "
              f"F1={train['f1']:.4f}; val acc={val['accuracy']:.4f} F1={val['f1']:.4f}", flush=True)
        if val["f1"] > best_f1:
            best_f1, best_epoch = val["f1"], epoch
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "classes": CLASSES, "size": args.size, "epoch": epoch,
                        "val_f1": best_f1, "config": config}, output / "best.pt")
    training_seconds = time.perf_counter() - start
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test = run_epoch(model, loaders["test"], device)
    test.update(best_epoch=best_epoch, best_val_f1=best_f1, training_seconds=training_seconds)
    (output / "test_metrics.json").write_text(json.dumps(test, indent=2))
    plot_history(history, output)
    plot_confusion(test["confusion_matrix"], output / "confusion_matrix.png", f"{args.model}: final test")
    print("FINAL TEST (validation-selected checkpoint):", flush=True)
    print(json.dumps(test, indent=2), flush=True)


if __name__ == "__main__":
    main()
