from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from .config import load_config, project_root_from_config, resolve_path
from .data import EDGE_TYPE_NAMES
from .metrics import evaluate_multilabel
from .model import ARHGNN
from .pipeline import prepare_data
from .train import predict_prob, set_seed, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARHGNN ablation experiments.")
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    args = parser.parse_args()
    run(args.config, args.epochs, args.bootstrap)


def run(config_path: str | Path, epochs_override: int | None = None, bootstrap_override: int | None = None) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    root = project_root_from_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    paths = config.get("paths", {})
    output_dir = resolve_path(root, paths.get("output_dir", "results_fixed/ablation"))
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_data(
        primary_path=resolve_path(root, paths.get("primary_data", "data/IgAdata.xlsx")),
        external_path=resolve_path(root, paths.get("external_data", "data/synthetic.xlsx")),
        seed=seed,
        test_size=float(config.get("split", {}).get("test_size", 0.2)),
        val_size=float(config.get("split", {}).get("val_size", 0.2)),
        use_smote=bool(config.get("imbalance", {}).get("use_smote", True)),
    )

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    epochs = int(epochs_override if epochs_override is not None else training_cfg.get("epochs", 60))
    bootstrap = int(bootstrap_override if bootstrap_override is not None else config.get("evaluation", {}).get("bootstrap", 300))
    threshold = float(config.get("evaluation", {}).get("threshold", 0.5))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variants = {
        "ARHGNN_full": {"use_attention": True, "use_residual": True},
        "no_attention": {"use_attention": False, "use_residual": True},
        "no_residual": {"use_attention": True, "use_residual": False},
        "plain_hgnn": {"use_attention": False, "use_residual": False},
    }

    summary_path = output_dir / "ablation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "best_epoch", "best_val_loss", "primary_macro_f1", "primary_macro_auc", "external_macro_f1", "external_macro_auc"])
        for name, flags in variants.items():
            set_seed(seed)
            model = ARHGNN(
                in_channels=prepared.x_train.shape[1],
                hidden_channels=int(model_cfg.get("hidden_channels", 64)),
                num_classes=prepared.y_train.shape[1],
                edge_type_count=len(EDGE_TYPE_NAMES),
                dropout=float(model_cfg.get("dropout", 0.5)),
                **flags,
            ).to(device)
            result = train_model(
                model,
                prepared,
                device,
                epochs=epochs,
                lr=float(training_cfg.get("learning_rate", 5e-4)),
                weight_decay=float(training_cfg.get("weight_decay", 5e-4)),
                patience=int(training_cfg.get("patience", 30)),
                loss_name="bce",
            )
            model.load_state_dict(result["best_state"])
            primary_prob = predict_prob(model, prepared.x_test, prepared.p_test, device)
            external_prob = predict_prob(model, prepared.x_external, prepared.p_external, device)
            primary_metrics = evaluate_multilabel(prepared.y_test, primary_prob, threshold, bootstrap, seed, prepared.primary.label_names)
            external_metrics = evaluate_multilabel(prepared.y_external, external_prob, threshold, bootstrap, seed, prepared.primary.label_names)
            writer.writerow(
                [
                    name,
                    result["best_epoch"],
                    result["best_val_loss"],
                    primary_metrics["summary"]["macro_f1"]["mean"],
                    primary_metrics["summary"]["macro_auc"]["mean"],
                    external_metrics["summary"]["macro_f1"]["mean"],
                    external_metrics["summary"]["macro_auc"]["mean"],
                ]
            )
    print(f"Saved ablation summary to {summary_path}")
    return summary_path


if __name__ == "__main__":
    main()

