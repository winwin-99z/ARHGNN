from __future__ import annotations

import argparse
from pathlib import Path

from .metrics import evaluate_multilabel, read_prediction_csv, write_metrics, write_per_label_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved ARHGNN prediction CSV.")
    parser.add_argument("--pred", required=True, help="CSV with True_Label_* and Prob_Class_* columns.")
    parser.add_argument("--out", default=None, help="Output metrics CSV path.")
    parser.add_argument("--per-label-out", default=None, help="Output per-label metrics CSV path.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    pred_path = Path(args.pred)
    y_true, y_prob, label_names = read_prediction_csv(pred_path)
    metrics = evaluate_multilabel(y_true, y_prob, args.threshold, args.bootstrap, label_names=label_names)
    out = Path(args.out) if args.out else pred_path.with_name(pred_path.stem.replace("predictions", "metrics") + ".csv")
    per_label_out = Path(args.per_label_out) if args.per_label_out else pred_path.with_name(pred_path.stem + "_per_label.csv")
    write_metrics(out, metrics)
    write_per_label_metrics(per_label_out, metrics)
    print(f"Saved metrics to {out}")
    print(f"Saved per-label metrics to {per_label_out}")


if __name__ == "__main__":
    main()

