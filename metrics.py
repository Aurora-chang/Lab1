"""Confusion matrix: rows=true class, columns=predicted class."""


def classification_metrics(matrix):
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp
    return {
        "accuracy": (tn + tp) / total if total else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "confusion_matrix": matrix,
        "samples": total,
    }
