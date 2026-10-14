import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import CLASSES


def plot_history(history, directory):
    epochs = [row["epoch"] for row in history]
    for metric in ("accuracy", "f1", "loss"):
        fig, ax = plt.subplots()
        for split in ("train", "val"):
            ax.plot(epochs, [row[f"{split}_{metric}"] for row in history], label=split)
        ax.set(xlabel="Epoch", ylabel=metric, title=f"Training and validation {metric}")
        if metric != "loss":
            ax.set_ylim(0, 1.02)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(directory / f"{metric}.png", dpi=160)
        plt.close(fig)


def plot_confusion(matrix, path, title="Final test confusion matrix"):
    fig, ax = plt.subplots()
    heatmap = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row][col]), ha="center", va="center",
                    color="white" if matrix[row][col] > max(map(max, matrix)) / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted class", ylabel="True class", title=title)
    fig.colorbar(heatmap, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
