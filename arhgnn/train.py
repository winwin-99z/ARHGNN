from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import load_config, project_root_from_config, resolve_path
from .data import EDGE_TYPE_NAMES
from .metrics import evaluate_multilabel, write_metrics, write_per_label_metrics, write_predictions
from .model import ARHGNN, FocalLoss
from .pipeline import PreparedData, prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ARHGNN for IgAN multi-label syndrome classification.")
    parser.add_argument("--config", default="configs/arhgnn.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args.config, epochs_override=args.epochs, bootstrap_override=args.bootstrap, device_override=args.device)


def run(
    config_path: str | Path,
    epochs_override: int | None = None,
    bootstrap_override: int | None = None,
    device_override: str | None = None,
) -> dict[str, object]:
    config_path = Path(config_path)
    config = load_config(config_path)
    root = project_root_from_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    paths = config.get("paths", {})
    output_dir = resolve_path(root, paths.get("output_dir", "results_fixed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    imbalance = config.get("imbalance", {})
    loss_name = str(config.get("training", {}).get("loss", "bce")).lower()
    use_smote = str(imbalance.get("method", "labelset_smote")).lower() == "labelset_smote"
    if loss_name == "focal" and use_smote:
        raise ValueError("Focal Loss sensitivity experiments must not be combined with SMOTE.")

    prepared = prepare_data(
        primary_path=resolve_path(root, paths.get("primary_data", "data/IgAdata.xlsx")),
        external_path=resolve_path(root, paths.get("external_data", "data/synthetic.xlsx")),
        seed=seed,
        test_size=float(config.get("split", {}).get("test_size", 0.2)),
        val_size=float(config.get("split", {}).get("val_size", 0.2)),
        use_smote=use_smote,
        categorical_max_unique=int(config.get("features", {}).get("categorical_max_unique", 8)),
        quantile_bins=int(config.get("hypergraph", {}).get("quantile_bins", 4)),
    )

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    epochs = int(epochs_override if epochs_override is not None else training_cfg.get("epochs", 100))
    bootstrap = int(bootstrap_override if bootstrap_override is not None else config.get("evaluation", {}).get("bootstrap", 1000))
    threshold = float(config.get("evaluation", {}).get("threshold", 0.5))
    device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = ARHGNN(
        in_channels=prepared.x_train.shape[1],
        hidden_channels=int(model_cfg.get("hidden_channels", 64)),
        num_classes=prepared.y_train.shape[1],
        edge_type_count=len(EDGE_TYPE_NAMES),
        dropout=float(model_cfg.get("dropout", 0.5)),
        use_attention=bool(model_cfg.get("use_attention", True)),
        use_residual=bool(model_cfg.get("use_residual", True)),
    ).to(device)

    train_result = train_model(
        model=model,
        prepared=prepared,
        device=device,
        epochs=epochs,
        lr=float(training_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 5e-4)),
        patience=int(training_cfg.get("patience", 50)),
        loss_name=loss_name,
    )

    checkpoint_path = output_dir / "best_arhgnn.pt"
    torch.save(
        {
            "model_state_dict": train_result["best_state"],
            "config": config,
            "label_names": prepared.primary.label_names,
            "encoded_feature_names": prepared.preprocessor.encoded_feature_names,
            "edge_type_names": EDGE_TYPE_NAMES,
        },
        checkpoint_path,
    )
    model.load_state_dict(train_result["best_state"])

    test_prob = predict_prob(model, prepared.x_test, prepared.p_test, device)
    external_prob = predict_prob(model, prepared.x_external, prepared.p_external, device)

    write_predictions(output_dir / "predictions_primary_test.csv", prepared.y_test, test_prob, prepared.primary.label_names, threshold)
    write_predictions(output_dir / "predictions_external.csv", prepared.y_external, external_prob, prepared.primary.label_names, threshold)

    test_metrics = evaluate_multilabel(prepared.y_test, test_prob, threshold, bootstrap, seed, prepared.primary.label_names)
    external_metrics = evaluate_multilabel(prepared.y_external, external_prob, threshold, bootstrap, seed, prepared.primary.label_names)
    write_metrics(output_dir / "metrics_primary_test.csv", test_metrics)
    write_metrics(output_dir / "metrics_external.csv", external_metrics)
    write_per_label_metrics(output_dir / "per_label_metrics_primary_test.csv", test_metrics)
    write_per_label_metrics(output_dir / "per_label_metrics_external.csv", external_metrics)

    dataset_info = {
        "primary_patients": prepared.primary.patient_count,
        "external_patients": prepared.external.patient_count,
        "label_names": prepared.primary.label_names,
        "primary_label_counts": prepared.primary.labels.sum(axis=0).astype(int).tolist(),
        "external_label_counts": prepared.external.labels.sum(axis=0).astype(int).tolist(),
        "primary_multilabel_rate": float((prepared.primary.labels.sum(axis=1) > 1).mean()),
        "external_multilabel_rate": float((prepared.external.labels.sum(axis=1) > 1).mean()),
        "split_sizes": {key: int(len(value)) for key, value in prepared.splits.items()},
        "smote_applied": prepared.smote_applied,
        "training_samples_after_smote": int(len(prepared.x_train)),
        "best_epoch": int(train_result["best_epoch"]),
        "best_val_loss": float(train_result["best_val_loss"]),
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "training_history.csv").write_text(train_result["history_csv"], encoding="utf-8-sig")

    try:
        from .plots import write_standard_plots

        write_standard_plots(output_dir, prepared.y_test, test_prob, prepared.y_external, external_prob, prepared.primary.label_names)
    except Exception as exc:
        (output_dir / "plot_generation_skipped.txt").write_text(str(exc), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "primary_test_metrics": test_metrics,
        "external_metrics": external_metrics,
    }


def train_model(
    model: ARHGNN,
    prepared: PreparedData,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    loss_name: str = "bce",
) -> dict[str, object]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion: nn.Module = FocalLoss() if loss_name == "focal" else nn.BCEWithLogitsLoss()

    x_train, p_train, y_train = _to_tensors(prepared.x_train, prepared.p_train, prepared.y_train, device)
    x_val, p_val, y_val = _to_tensors(prepared.x_val, prepared.p_val, prepared.y_val, device)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history_rows = ["epoch,train_loss,val_loss\n"]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train, p_train)
        train_loss = criterion(logits, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val, p_val)
            val_loss = criterion(val_logits, y_val).item()
        history_rows.append(f"{epoch},{train_loss.item():.8f},{val_loss:.8f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best model state.")
    return {
        "best_state": best_state,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history_csv": "".join(history_rows),
    }


def predict_prob(model: ARHGNN, x: np.ndarray, propagation: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    p_tensor = torch.tensor(propagation, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x_tensor, p_tensor)
        return torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)


def _to_tensors(
    x: np.ndarray,
    propagation: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(x, dtype=torch.float32, device=device),
        torch.tensor(propagation, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()

