from __future__ import annotations

from pathlib import Path

import numpy as np


def write_standard_plots(
    output_dir: str | Path,
    y_test: np.ndarray,
    p_test: np.ndarray,
    y_external: np.ndarray,
    p_external: np.ndarray,
    label_names: list[str],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    _plot_roc(output_dir / "roc_primary_test.png", y_test, p_test, label_names, "Primary Test ROC")
    _plot_roc(output_dir / "roc_external.png", y_external, p_external, label_names, "External Validation ROC")
    _plot_calibration(output_dir / "calibration_external.png", y_external, p_external, label_names)
    _plot_dca(output_dir / "dca_external.png", y_external, p_external, label_names)


def _plot_roc(path: Path, y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str], title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, label in enumerate(label_names):
        fpr, tpr = _roc_curve(y_true[:, i], y_prob[:, i])
        if fpr is None:
            continue
        ax.plot(fpr, tpr, lw=1.8, label=label)
    ax.plot([0, 1], [0, 1], color="0.4", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_calibration(path: Path, y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str], bins: int = 8) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    grid = np.linspace(0, 1, bins + 1)
    for i, label in enumerate(label_names):
        observed = []
        predicted = []
        for left, right in zip(grid[:-1], grid[1:]):
            mask = (y_prob[:, i] >= left) & (y_prob[:, i] < right if right < 1 else y_prob[:, i] <= right)
            if mask.sum() == 0:
                continue
            observed.append(float(y_true[mask, i].mean()))
            predicted.append(float(y_prob[mask, i].mean()))
        ax.plot(predicted, observed, marker="o", lw=1.5, label=label)
    ax.plot([0, 1], [0, 1], color="0.4", linestyle="--", lw=1)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("External Validation Calibration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_dca(path: Path, y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str]) -> None:
    import matplotlib.pyplot as plt

    thresholds = np.linspace(0.01, 0.99, 99)
    fig, axes = plt.subplots(1, len(label_names), figsize=(3.2 * len(label_names), 3.2), sharey=True)
    if len(label_names) == 1:
        axes = [axes]
    for i, (ax, label) in enumerate(zip(axes, label_names)):
        model_nb = [_net_benefit(y_true[:, i], y_prob[:, i] >= t, t) for t in thresholds]
        treat_all = [_net_benefit(y_true[:, i], np.ones_like(y_true[:, i], dtype=bool), t) for t in thresholds]
        ax.plot(thresholds, model_nb, label="Model", lw=1.7)
        ax.plot(thresholds, treat_all, label="Treat all", lw=1.2)
        ax.axhline(0, color="0.4", linestyle="--", lw=1)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Threshold")
    axes[0].set_ylabel("Net Benefit")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    y_true = y_true.astype(int)
    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        return None, None
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tpr = np.concatenate([[0.0], np.cumsum(y_sorted == 1) / pos, [1.0]])
    fpr = np.concatenate([[0.0], np.cumsum(y_sorted == 0) / neg, [1.0]])
    return fpr, tpr


def _net_benefit(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = np.logical_and(y_true, y_pred).sum()
    fp = np.logical_and(~y_true, y_pred).sum()
    n = len(y_true)
    return float((tp / n) - (fp / n) * (threshold / (1 - threshold)))

