"""Offline synthetic checks; these are software tests, not lab experiments."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from PIL import Image

from dataset import build_loaders, CLASSES
from metrics import classification_metrics
from model import build_model


class PipelineTests(unittest.TestCase):
    def test_positive_class_metrics_and_zero_denominators(self):
        result = classification_metrics([[8, 2], [3, 7]])
        self.assertEqual(result["samples"], 20)
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["f1"], 14 / 19)
        self.assertAlmostEqual(result["precision"], 7 / 9)
        self.assertAlmostEqual(result["recall"], 0.7)
        self.assertEqual(classification_metrics([[5, 0], [0, 0]])["f1"], 0)

    def test_offline_training_checkpoint_inference_and_comparison(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "data"
            for split, count in (("train", 8), ("test", 2)):
                for label, name in enumerate(CLASSES):
                    folder = dataset / split / name
                    folder.mkdir(parents=True)
                    for index in range(count):
                        Image.new("RGB", (64, 64), (30 + label * 140, index * 20, 70)).save(folder / f"{index}.png")
            kwargs = dict(root=dataset, batch_size=4, size=64, workers=0, seed=42)
            _, samples = build_loaders(**kwargs)
            _, repeat = build_loaders(**kwargs)
            self.assertEqual(samples, repeat)
            path_sets = [set(path for path, _ in samples[split]) for split in ("train", "val", "test")]
            self.assertFalse(path_sets[0] & path_sets[1])
            self.assertFalse((path_sets[0] | path_sets[1]) & path_sets[2])
            for split in ("train", "val", "test"):
                self.assertEqual(set(label for _, label in samples[split]), {0, 1})
            runs = root / "results"
            for name in ("resnet18", "resnet50"):
                subprocess.run([sys.executable, "train.py", "--dataset", str(dataset),
                                "--model", name, "--epochs", "1", "--batch-size", "4",
                                "--size", "64", "--workers", "0", "--no-pretrained",
                                "--device", "cpu", "--output", str(runs)], check=True)
                run = runs / name
                metrics = json.loads((run / "test_metrics.json").read_text())
                self.assertEqual(metrics["samples"], 4)
                for filename in ("best.pt", "history.csv", "accuracy.png", "f1.png", "confusion_matrix.png"):
                    self.assertTrue((run / filename).is_file())
                checkpoint = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
                model = build_model(name, pretrained=False)
                model.load_state_dict(checkpoint["state_dict"])
                evaluation = root / f"evaluation_{name}"
                subprocess.run([sys.executable, "inference.py", "--checkpoint", str(run / "best.pt"),
                                "--test-dir", str(dataset / "test"), "--device", "cpu",
                                "--output", str(evaluation)], check=True)
                reloaded = json.loads((evaluation / "metrics.json").read_text())
                self.assertEqual(metrics["confusion_matrix"], reloaded["confusion_matrix"])
            prediction = subprocess.run([sys.executable, "inference.py", "--checkpoint", str(runs / "resnet18/best.pt"),
                                         "--image", str(dataset / "test/NORMAL/0.png"), "--device", "cpu"],
                                        check=True, capture_output=True, text=True)
            self.assertIn(json.loads(prediction.stdout)["prediction"], CLASSES)
            comparison = root / "comparison"
            subprocess.run([sys.executable, "compare.py", "--runs", str(runs / "resnet18"),
                            str(runs / "resnet50"), "--output", str(comparison)], check=True)
            self.assertTrue((comparison / "summary.csv").exists())
            self.assertTrue((comparison / "val_f1.png").exists())


if __name__ == "__main__":
    unittest.main()
