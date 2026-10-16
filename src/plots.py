"""Matplotlib figures required by the assignment."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_residuals(y_true, y_pred, path, code):
    res = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, res, alpha=0.7)
    plt.axhline(0, color="red")
    plt.xlabel("Predicted yield (tons)"); plt.ylabel("Residual (actual - predicted)")
    plt.title(f"Residuals [{code}]"); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_confusion(cm, path, code):
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["Not high", "High risk"]); plt.yticks([0, 1], ["Not high", "High risk"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Confusion matrix [{code}]"); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def plot_clusters(coords, labels, path, code):
    plt.figure(figsize=(6, 4.5))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis", alpha=0.8)
    plt.legend(*sc.legend_elements(), title="Cluster")
    plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
    plt.title(f"Farm clusters [{code}]"); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
