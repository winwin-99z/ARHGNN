from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import (
    ClinicalDataset,
    FeaturePreprocessor,
    PatientHypergraphBuilder,
    load_dataset,
    validate_no_label_leakage,
)
from .smote import labelset_smote
from .splits import split_train_val_test


@dataclass
class PreparedData:
    primary: ClinicalDataset
    external: ClinicalDataset
    preprocessor: FeaturePreprocessor
    builder: PatientHypergraphBuilder
    splits: dict[str, np.ndarray]
    x_train: np.ndarray
    y_train: np.ndarray
    p_train: np.ndarray
    x_val_graph: np.ndarray
    y_val: np.ndarray
    p_val: np.ndarray
    val_query_start: int
    x_test_graph: np.ndarray
    y_test: np.ndarray
    p_test: np.ndarray
    test_query_start: int
    x_external_graph: np.ndarray
    y_external: np.ndarray
    p_external: np.ndarray
    external_query_start: int
    smote_applied: bool


def prepare_data(
    primary_path: str | Path,
    external_path: str | Path,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
    use_smote: bool = True,
    categorical_max_unique: int = 8,
    quantile_bins: int = 4,
) -> PreparedData:
    primary = load_dataset(primary_path)
    external = load_dataset(external_path, label_columns=primary.label_names)
    validate_no_label_leakage(primary.feature_names, primary.label_names)
    validate_no_label_leakage(external.feature_names, external.label_names)

    splits = split_train_val_test(primary.labels, test_size=test_size, val_size=val_size, seed=seed)
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    preprocessor = FeaturePreprocessor(categorical_max_unique=categorical_max_unique)
    preprocessor.fit(primary.raw_features.iloc[train_idx])
    x_primary = preprocessor.transform(primary.raw_features)
    x_external = preprocessor.transform(external.raw_features)

    x_train = x_primary[train_idx]
    y_train = primary.labels[train_idx]
    smote_applied = False
    if use_smote:
        x_train, y_train = labelset_smote(x_train, y_train, seed=seed)
        smote_applied = len(x_train) != len(train_idx)

    builder = PatientHypergraphBuilder(quantile_bins=quantile_bins)
    builder.fit(x_train, preprocessor.encoded_feature_names, preprocessor.encoded_feature_groups)

    train_hg = builder.transform(x_train)
    x_val_query = x_primary[val_idx]
    x_test_query = x_primary[test_idx]
    # Held-out and external patients are attached only to fixed training nodes.
    val_hg = builder.transform_inductive(x_train, x_val_query)
    test_hg = builder.transform_inductive(x_train, x_test_query)
    external_hg = builder.transform_inductive(x_train, x_external)

    return PreparedData(
        primary=primary,
        external=external,
        preprocessor=preprocessor,
        builder=builder,
        splits=splits,
        x_train=x_train,
        y_train=y_train,
        p_train=train_hg.propagation,
        x_val_graph=np.vstack([x_train, x_val_query]).astype(np.float32),
        y_val=primary.labels[val_idx],
        p_val=val_hg.propagation,
        val_query_start=int(len(x_train)),
        x_test_graph=np.vstack([x_train, x_test_query]).astype(np.float32),
        y_test=primary.labels[test_idx],
        p_test=test_hg.propagation,
        test_query_start=int(len(x_train)),
        x_external_graph=np.vstack([x_train, x_external]).astype(np.float32),
        y_external=external.labels,
        p_external=external_hg.propagation,
        external_query_start=int(len(x_train)),
        smote_applied=smote_applied,
    )
