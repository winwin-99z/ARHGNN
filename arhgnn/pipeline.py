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
    x_val: np.ndarray
    y_val: np.ndarray
    p_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    p_test: np.ndarray
    x_external: np.ndarray
    y_external: np.ndarray
    p_external: np.ndarray
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
    val_hg = builder.transform(x_primary[val_idx])
    test_hg = builder.transform(x_primary[test_idx])
    external_hg = builder.transform(x_external)

    return PreparedData(
        primary=primary,
        external=external,
        preprocessor=preprocessor,
        builder=builder,
        splits=splits,
        x_train=x_train,
        y_train=y_train,
        p_train=train_hg.propagation,
        x_val=x_primary[val_idx],
        y_val=primary.labels[val_idx],
        p_val=val_hg.propagation,
        x_test=x_primary[test_idx],
        y_test=primary.labels[test_idx],
        p_test=test_hg.propagation,
        x_external=x_external,
        y_external=external.labels,
        p_external=external_hg.propagation,
        smote_applied=smote_applied,
    )

