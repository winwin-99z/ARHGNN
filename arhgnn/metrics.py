from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-logits))


def _labelset_stratified_bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample patients within observed multi-label strata.

    Each evaluation call contains one cohort or split.  Keeping the number of
    patients in every observed label-set stratum fixed preserves the cohort's
    multi-label case mix while resampling patients with replacement within it.
    """
    label_sets = np.asarray(y_true, dtype=np.int8)
    _, inverse = np.unique(label_sets, axis=0, return_inverse=True)
    sampled = []
    for stratum in range(int(inverse.max()) + 1):
        members = np.flatnonzero(inverse == stratum)
        sampled.append(rng.choice(members, size=len(members), replace=True))
    return np.concatenate(sampled)


def evaluate_multilabel(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    bootstrap: int = 1000,
    seed: int = 42,
    label_names: list[str] | None = None,
    common_calibration_bins: int = 10,
    sparse_calibration_bins: int = 5,
    sparse_positive_cutoff: int = 20,
) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}")
    if label_names is None:
        label_names = [f"label_{i}" for i in range(y_true.shape[1])]

    summary, per_label = _compute_metrics(
        y_true, y_prob, threshold, label_names,
        common_calibration_bins, sparse_calibration_bins, sparse_positive_cutoff,
    )
    if bootstrap > 0 and len(y_true) > 1:
        rng = np.random.default_rng(seed)
        boot_values: dict[str, list[float]] = {key: [] for key in summary}
        for _ in range(bootstrap):
            idx = _labelset_stratified_bootstrap_indices(y_true, rng)
            boot_summary, _ = _compute_metrics(
                y_true[idx], y_prob[idx], threshold, label_names,
                common_calibration_bins, sparse_calibration_bins, sparse_positive_cutoff,
            )
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
        "bootstrap": {
            "iterations": int(bootstrap),
            "method": "patient_level_labelset_stratified_within_evaluation_cohort",
        },
        "calibration": {
            "binning": "equal_frequency",
            "common_label_bins": int(common_calibration_bins),
            "sparse_label_bins": int(sparse_calibration_bins),
            "sparse_positive_cutoff": int(sparse_positive_cutoff),
        },
        "summary": summary_with_ci,
        "per_label": per_label,
    }


def write_predictions(
    path: str | Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    threshold: float = 0.5,
    cohort_id: str | None = None,
    split_id: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = ["Record_ID", "Cohort_ID", "Split_ID"]
        header.extend([f"True_Label_{name}" for name in label_names])
        header.extend([f"Prob_Class_{name}" for name in label_names])
        header.extend([f"Pred_Label_{name}" for name in label_names])
        writer.writerow(header)
        for index, (true_row, prob_row, pred_row) in enumerate(zip(y_true.astype(int), y_prob, y_pred), start=1):
            writer.writerow(
                [f"record_{index:06d}", cohort_id or "unspecified", split_id or "unspecified"]
                + true_row.tolist()
                + [float(v) for v in prob_row]
                + pred_row.tolist()
            )


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


def metric_invariants(metrics: dict[str, object]) -> dict[str, float | bool]:
    """Return machine-readable checks linking aggregate metrics to per-label counts."""

    summary = metrics["summary"]
    per_label = metrics["per_label"]
    accuracy = float(summary["elementwise_accuracy"]["mean"])
    hamming = float(summary["hamming_loss"]["mean"])
    supports = np.asarray([row["positives"] for row in per_label], dtype=float)
    f1 = np.asarray([row["f1"] for row in per_label], dtype=float)
    weighted_f1 = float(summary["weighted_f1"]["mean"])
    recomputed_weighted_f1 = float(np.sum(supports * f1) / supports.sum()) if supports.sum() else 0.0
    correct = int(sum(int(row["tp"]) + int(row["tn"]) for row in per_label))
    total = int(sum(int(row["n"]) for row in per_label))
    return {
        "accuracy_plus_hamming_equals_one": bool(np.isclose(accuracy + hamming, 1.0, atol=1e-12)),
        "weighted_f1_reproducible_from_per_label_values": bool(
            np.isclose(weighted_f1, recomputed_weighted_f1, atol=1e-12)
        ),
        "figure8_tally_matches_elementwise_accuracy": bool(
            total > 0 and np.isclose(correct / total, accuracy, atol=1e-12)
        ),
        "correct_patient_label_decisions": correct,
        "total_patient_label_decisions": total,
        "weighted_f1_from_per_label_values": recomputed_weighted_f1,
    }


def write_metric_audit(path: str | Path, metrics: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": metrics["threshold"],
        "n_samples": metrics["n_samples"],
        "calibration": metrics["calibration"],
        "invariants": metric_invariants(metrics),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    label_names: list[str],
    common_calibration_bins: int,
    sparse_calibration_bins: int,
    sparse_positive_cutoff: int,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    y_pred = (y_prob >= threshold).astype(np.float32)
    tp = ((y_true == 1) & (y_pred == 1)).sum(axis=0).astype(float)
    fp = ((y_true == 0) & (y_pred == 1)).sum(axis=0).astype(float)
    fn = ((y_true == 1) & (y_pred == 0)).sum(axis=0).astype(float)
    tn = ((y_true == 0) & (y_pred == 0)).sum(axis=0).astype(float)
    support = y_true.sum(axis=0).astype(float)
    negatives = y_true.shape[0] - support

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
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
        precision_ci_low, precision_ci_high = _wilson_ci(int(tp[i]), int(tp[i] + fp[i]))
        recall_ci_low, recall_ci_high = _wilson_ci(int(tp[i]), int(support[i]))
        specificity_ci_low, specificity_ci_high = _wilson_ci(int(tn[i]), int(tn[i] + fp[i]))
        bin_count = sparse_calibration_bins if support[i] < sparse_positive_cutoff else common_calibration_bins
        brier, ece, occupied_bins = _calibration_metrics(y_true[:, i], y_prob[:, i], bin_count)
        per_label.append(
            {
                "label": name,
                "n": int(y_true.shape[0]),
                "support": int(support[i]),
                "positives": int(support[i]),
                "negatives": int(negatives[i]),
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "tn": int(tn[i]),
                "precision": float(precision[i]),
                "precision_ci_low": precision_ci_low,
                "precision_ci_high": precision_ci_high,
                "recall": float(recall[i]),
                "recall_ci_low": recall_ci_low,
                "recall_ci_high": recall_ci_high,
                "specificity": float(specificity[i]),
                "specificity_ci_low": specificity_ci_low,
                "specificity_ci_high": specificity_ci_high,
                "f1": float(f1[i]),
                "auc": float(aucs[i]),
                "auprc": float(aps[i]),
                "average_precision": float(aps[i]),
                "brier_score": brier,
                "expected_calibration_error": ece,
                "calibration_bin_count": int(bin_count),
                "occupied_bins": int(occupied_bins),
            }
        )

    hamming_loss = float(np.mean(y_true != y_pred))
    summary = {
        "elementwise_accuracy": float(np.mean(y_true == y_pred)),
        "hamming_loss": hamming_loss,
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


def _calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, bin_count: int) -> tuple[float, float, int]:
    """Brier score and ECE using non-empty equal-frequency probability bins."""

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        return float("nan"), float("nan"), 0
    bins = [group for group in np.array_split(np.argsort(y_prob, kind="stable"), max(1, int(bin_count))) if len(group)]
    ece = 0.0
    for indices in bins:
        observed = float(y_true[indices].mean())
        predicted = float(y_prob[indices].mean())
        ece += len(indices) / len(y_true) * abs(observed - predicted)
    return float(np.mean((y_prob - y_true) ** 2)), float(ece), len(bins)


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
    return float(np.trapz(tpr, fpr))


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
