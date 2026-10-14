"""Create model comparison figures from completed runs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/comparison"))
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("Provide at least two completed runs")
    runs = []
    for directory in args.runs:
        config = json.loads((directory / "config.json").read_text())
        history = list(csv.DictReader((directory / "history.csv").open()))
        metrics = json.loads((directory / "test_metrics.json").read_text())
        runs.append((config["model"], history, metrics))
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        for metric in ("accuracy", "f1"):
            fig, ax = plt.subplots()
            for name, history, _ in runs:
                ax.plot([int(row["epoch"]) for row in history],
                        [float(row[f"{split}_{metric}"]) for row in history], label=name)
            ax.set(xlabel="Epoch", ylabel=metric, ylim=(0, 1.02), title=f"{split} {metric} comparison")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(args.output / f"{split}_{metric}.png", dpi=160)
            plt.close(fig)
    with (args.output / "summary.csv").open("w", newline="") as stream:
        fields = ["model", "accuracy", "f1", "precision", "recall", "best_epoch", "training_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, _, metrics in runs:
            writer.writerow({"model": name, **{key: metrics[key] for key in fields[1:]}})
            print(f"{name}: test accuracy={metrics['accuracy']:.2%}, F1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
