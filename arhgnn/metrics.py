from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits))


def evaluate_multilabel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    bootstrap: int = 1000,
    seed: int = 42,
    label_names: list[str] | None = None,
) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}")
    if label_names is None:
        label_names = [f"label_{i}" for i in range(y_true.shape[1])]

    summary, per_label = _compute_metrics(y_true, y_prob, threshold, label_names)
    if bootstrap > 0 and len(y_true) > 1:
        rng = np.random.default_rng(seed)
        boot_values: dict[str, list[float]] = {key: [] for key in summary}
        for _ in range(bootstrap):
            idx = rng.integers(0, len(y_true), size=len(y_true))
            boot_summary, _ = _compute_metrics(y_true[idx], y_prob[idx], threshold, label_names)
            for key, value in boot_summary.items():
                boot_values[key].append(value)
        summary_with_ci = {}
        for key, value in summary.items():
            values = np.asarray(boot_values[key], dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                ci_low = ci_high = float("nan")
            else:
                ci_low, ci_high = np.percentile(values, [2.5, 97.5])
            summary_with_ci[key] = {"mean": float(value), "ci_low": float(ci_low), "ci_high": float(ci_high)}
    else:
        summary_with_ci = {
            key: {"mean": float(value), "ci_low": float("nan"), "ci_high": float("nan")}
            for key, value in summary.items()
        }

    return {
        "threshold": float(threshold),
        "n_samples": int(y_true.shape[0]),
        "summary": summary_with_ci,
        "per_label": per_label,
    }


def write_predictions(
    path: str | Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    threshold: float = 0.5,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = []
        header.extend([f"True_Label_{name}" for name in label_names])
        header.extend([f"Prob_Class_{name}" for name in label_names])
        header.extend([f"Pred_Label_{name}" for name in label_names])
        writer.writerow(header)
        for true_row, prob_row, pred_row in zip(y_true.astype(int), y_prob, y_pred):
            writer.writerow(true_row.tolist() + [float(v) for v in prob_row] + pred_row.tolist())


def read_prediction_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import pandas as pd

    df = pd.read_csv(path)
    true_cols = [c for c in df.columns if c.startswith("True_Label_")]
    prob_cols = [c for c in df.columns if c.startswith("Prob_Class_")]
    if not true_cols or not prob_cols:
        raise ValueError("Prediction CSV must contain True_Label_* and Prob_Class_* columns.")
    label_names = [c.replace("True_Label_", "", 1) for c in true_cols]
    prob_names = [c.replace("Prob_Class_", "", 1) for c in prob_cols]
    if label_names != prob_names:
        raise ValueError("True label columns and probability columns use different label order.")
    return (
        df[true_cols].to_numpy(dtype=np.float32),
        df[prob_cols].to_numpy(dtype=np.float32),
        label_names,
    )


def write_metrics(path: str | Path, metrics: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "ci_low", "ci_high"])
        for key, values in metrics["summary"].items():
            writer.writerow([key, values["mean"], values["ci_low"], values["ci_high"]])


def write_per_label_metrics(path: str | Path, metrics: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = metrics["per_label"]
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    label_names: list[str],
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    y_pred = (y_prob >= threshold).astype(np.float32)
    tp = ((y_true == 1) & (y_pred == 1)).sum(axis=0).astype(float)
    fp = ((y_true == 0) & (y_pred == 1)).sum(axis=0).astype(float)
    fn = ((y_true == 1) & (y_pred == 0)).sum(axis=0).astype(float)
    support = y_true.sum(axis=0).astype(float)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    weights = support / support.sum() if support.sum() > 0 else np.zeros_like(support)

    micro_tp, micro_fp, micro_fn = tp.sum(), fp.sum(), fn.sum()
    micro_precision = _safe_scalar_div(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_scalar_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = _safe_scalar_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    aucs = np.asarray([_binary_auc(y_true[:, i], y_prob[:, i]) for i in range(y_true.shape[1])], dtype=float)
    aps = np.asarray([_average_precision(y_true[:, i], y_prob[:, i]) for i in range(y_true.shape[1])], dtype=float)

    per_label = []
    for i, name in enumerate(label_names):
        ci_low, ci_high = _wilson_ci(int(tp[i]), int(support[i]))
        per_label.append(
            {
                "label": name,
                "support": int(support[i]),
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "recall_ci_low": ci_low,
                "recall_ci_high": ci_high,
                "f1": float(f1[i]),
                "auc": float(aucs[i]),
                "average_precision": float(aps[i]),
            }
        )

    summary = {
        "hamming_loss": float(np.mean(y_true != y_pred)),
        "subset_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "macro_precision": float(np.nanmean(precision)),
        "macro_recall": float(np.nanmean(recall)),
        "macro_f1": float(np.nanmean(f1)),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "weighted_f1": float(np.nansum(weights * f1)),
        "macro_auc": float(np.nanmean(aucs)),
        "macro_ap": float(np.nanmean(aps)),
    }
    return summary, per_label


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(num, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def _safe_scalar_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    positives = y_true == 1
    negatives = y_true == 0
    if positives.sum() == 0 or negatives.sum() == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0.0], tps / positives.sum(), [1.0]])
    fpr = np.concatenate([[0.0], fps / negatives.sum(), [1.0]])
    return float(np.trapezoid(tpr, fpr))


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    positives = int((y_true == 1).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp_cumsum = np.cumsum(y_sorted == 1)
    precision_at_k = tp_cumsum / (np.arange(len(y_sorted)) + 1)
    return float((precision_at_k * (y_sorted == 1)).sum() / positives)


def _wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return float((centre - margin) / denom), float((centre + margin) / denom)
