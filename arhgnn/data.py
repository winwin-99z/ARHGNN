from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_LABEL_COUNT = 5
EDGE_TYPE_NAMES = (
    "demographics",
    "primary_symptom",
    "secondary_symptom",
    "tongue",
    "pulse",
    "other",
)


@dataclass(frozen=True)
class ClinicalDataset:
    raw_features: pd.DataFrame
    labels: np.ndarray
    feature_names: list[str]
    label_names: list[str]
    patient_count: int
    source_path: str


@dataclass(frozen=True)
class HypergraphData:
    propagation: np.ndarray
    edge_count: int
    edge_metadata: list[dict[str, object]]
    edge_type_names: tuple[str, ...] = EDGE_TYPE_NAMES


def load_dataset(path: str | Path, label_columns: Iterable[str] | None = None) -> ClinicalDataset:
    """Load an IgAN cohort and split clinical descriptors from the five labels.

    If ``label_columns`` is omitted, the final five columns are treated as labels.
    This mirrors the source spreadsheets and prevents label columns from entering
    feature preprocessing or hypergraph construction.
    """

    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")

    if label_columns is None:
        label_names = list(df.columns[-DEFAULT_LABEL_COUNT:])
    else:
        label_names = list(label_columns)

    missing = [name for name in label_names if name not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns in {path}: {missing}")

    feature_df = df.drop(columns=label_names).copy()
    label_df = df[label_names].copy()
    labels = label_df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    unique_label_values = set(np.unique(labels).tolist())
    if not unique_label_values.issubset({0.0, 1.0}):
        raise ValueError(f"Labels must be binary 0/1 values; got {sorted(unique_label_values)}")

    return ClinicalDataset(
        raw_features=feature_df,
        labels=labels,
        feature_names=list(feature_df.columns),
        label_names=label_names,
        patient_count=int(len(df)),
        source_path=str(path),
    )


def infer_feature_group(name: str) -> str:
    text = str(name)
    if "年龄" in text or "性别" in text:
        return "demographics"
    if "舌" in text:
        return "tongue"
    if "脉" in text:
        return "pulse"
    primary_tokens = (
        "尿蛋白",
        "血尿",
        "红细胞",
        "头晕",
        "血压",
        "收缩压",
        "舒张压",
        "肾功能",
        "肾小球",
        "GFR",
        "口气",
        "尿臭",
    )
    if any(token in text for token in primary_tokens):
        return "primary_symptom"
    return "secondary_symptom"


class FeaturePreprocessor:
    """Train-only fitted feature encoder for mixed clinical spreadsheets."""

    def __init__(self, categorical_max_unique: int = 8, continuous_eps: float = 1e-6):
        self.categorical_max_unique = categorical_max_unique
        self.continuous_eps = continuous_eps
        self.feature_specs: list[dict[str, object]] = []
        self.encoded_feature_names: list[str] = []
        self.encoded_feature_groups: list[str] = []
        self.raw_feature_names: list[str] = []

    def fit(self, features: pd.DataFrame) -> "FeaturePreprocessor":
        self.raw_feature_names = list(features.columns)
        self.feature_specs = []
        self.encoded_feature_names = []
        self.encoded_feature_groups = []

        for column in self.raw_feature_names:
            series = features[column]
            group = infer_feature_group(column)
            numeric = pd.api.types.is_numeric_dtype(series)
            unique_count = int(series.dropna().nunique())

            if numeric and unique_count > self.categorical_max_unique:
                values = pd.to_numeric(series, errors="coerce")
                mean = float(values.mean()) if not np.isnan(values.mean()) else 0.0
                std = float(values.std(ddof=0)) if not np.isnan(values.std(ddof=0)) else 1.0
                if std < self.continuous_eps:
                    std = 1.0
                self.feature_specs.append(
                    {
                        "column": column,
                        "kind": "continuous",
                        "mean": mean,
                        "std": std,
                        "group": group,
                    }
                )
                self.encoded_feature_names.append(column)
                self.encoded_feature_groups.append(group)
            else:
                categories = sorted({_normalize_category(value) for value in series})
                if "__MISSING__" not in categories:
                    categories.append("__MISSING__")
                if "__UNKNOWN__" not in categories:
                    categories.append("__UNKNOWN__")
                self.feature_specs.append(
                    {
                        "column": column,
                        "kind": "categorical",
                        "categories": categories,
                        "group": group,
                    }
                )
                for category in categories:
                    self.encoded_feature_names.append(f"{column}={category}")
                    self.encoded_feature_groups.append(group)
        return self

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        if not self.feature_specs:
            raise RuntimeError("FeaturePreprocessor must be fitted before transform().")
        missing = [name for name in self.raw_feature_names if name not in features.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        encoded_columns: list[np.ndarray] = []
        for spec in self.feature_specs:
            column = str(spec["column"])
            if spec["kind"] == "continuous":
                values = pd.to_numeric(features[column], errors="coerce").to_numpy(dtype=np.float32)
                values = np.where(np.isnan(values), float(spec["mean"]), values)
                encoded_columns.append(((values - float(spec["mean"])) / float(spec["std"]))[:, None])
            else:
                categories = list(spec["categories"])
                index = {category: i for i, category in enumerate(categories)}
                arr = np.zeros((len(features), len(categories)), dtype=np.float32)
                for row, value in enumerate(features[column]):
                    category = _normalize_category(value)
                    if category not in index:
                        category = "__UNKNOWN__"
                    arr[row, index[category]] = 1.0
                encoded_columns.append(arr)
        return np.concatenate(encoded_columns, axis=1).astype(np.float32)

    def fit_transform(self, features: pd.DataFrame) -> np.ndarray:
        return self.fit(features).transform(features)


class PatientHypergraphBuilder:
    """Build patient-level hypergraph propagation matrices from clinical features only."""

    def __init__(self, quantile_bins: int = 4, min_edge_size: int = 1):
        self.quantile_bins = quantile_bins
        self.min_edge_size = min_edge_size
        self.feature_specs: list[dict[str, object]] = []
        self.feature_names: list[str] = []
        self.feature_groups: list[str] = []
        self.fitted = False

    def fit(
        self,
        x_features: np.ndarray,
        feature_names: Iterable[str] | None = None,
        feature_groups: Iterable[str] | None = None,
    ) -> "PatientHypergraphBuilder":
        x = _ensure_2d_float(x_features)
        n_features = x.shape[1]
        self.feature_names = list(feature_names) if feature_names is not None else [f"feature_{i}" for i in range(n_features)]
        self.feature_groups = list(feature_groups) if feature_groups is not None else ["other"] * n_features
        if len(self.feature_names) != n_features:
            raise ValueError("feature_names length does not match x_features width.")
        if len(self.feature_groups) != n_features:
            raise ValueError("feature_groups length does not match x_features width.")

        self.feature_specs = []
        for col in range(n_features):
            values = x[:, col]
            finite = values[np.isfinite(values)]
            unique = np.unique(np.round(finite, 8))
            binary_like = len(unique) <= 2 and set(unique.tolist()).issubset({0.0, 1.0})
            if binary_like:
                self.feature_specs.append({"kind": "binary", "column": col})
            else:
                quantiles = np.linspace(0, 1, self.quantile_bins + 1)[1:-1]
                if len(finite) == 0:
                    edges = np.array([], dtype=np.float32)
                else:
                    edges = np.unique(np.quantile(finite, quantiles)).astype(np.float32)
                self.feature_specs.append({"kind": "continuous", "column": col, "edges": edges})
        self.fitted = True
        return self

    def transform(self, x_features: np.ndarray) -> HypergraphData:
        if not self.fitted:
            raise RuntimeError("PatientHypergraphBuilder must be fitted before transform().")
        x = _ensure_2d_float(x_features)
        n_patients, n_features = x.shape
        if n_features != len(self.feature_specs):
            raise ValueError("x_features width differs from fitted feature count.")

        type_count = len(EDGE_TYPE_NAMES)
        accum = np.zeros((type_count, n_patients, n_patients), dtype=np.float32)
        degrees = np.zeros((type_count, n_patients), dtype=np.float32)
        metadata: list[dict[str, object]] = []

        for spec, name, group in zip(self.feature_specs, self.feature_names, self.feature_groups):
            col = int(spec["column"])
            type_id = EDGE_TYPE_NAMES.index(group) if group in EDGE_TYPE_NAMES else EDGE_TYPE_NAMES.index("other")
            values = x[:, col]
            if spec["kind"] == "binary":
                edge_groups = [(f"{name}=active", np.where(values > 0.5)[0])]
            else:
                bins = np.digitize(values, np.asarray(spec["edges"], dtype=np.float32), right=False)
                edge_groups = [(f"{name}=bin_{bin_id}", np.where(bins == bin_id)[0]) for bin_id in np.unique(bins)]

            for edge_name, nodes in edge_groups:
                nodes = np.asarray(nodes, dtype=np.int64)
                if len(nodes) < self.min_edge_size:
                    continue
                weight = 1.0 / float(len(nodes))
                accum[type_id][np.ix_(nodes, nodes)] += weight
                degrees[type_id, nodes] += 1.0
                metadata.append({"name": edge_name, "type": EDGE_TYPE_NAMES[type_id], "size": int(len(nodes))})

        total_degrees = degrees.sum(axis=0)
        isolated = np.where(total_degrees == 0)[0]
        if len(isolated) > 0:
            other = EDGE_TYPE_NAMES.index("other")
            accum[other, isolated, isolated] += 1.0
            degrees[other, isolated] += 1.0
            for node in isolated:
                metadata.append({"name": f"self_loop_{node}", "type": "other", "size": 1})

        for type_id in range(type_count):
            deg = degrees[type_id]
            nonzero = deg > 0
            scale = np.zeros_like(deg, dtype=np.float32)
            scale[nonzero] = 1.0 / np.sqrt(deg[nonzero])
            accum[type_id] = accum[type_id] * scale[:, None] * scale[None, :]

        return HypergraphData(
            propagation=accum.astype(np.float32),
            edge_count=len(metadata),
            edge_metadata=metadata,
        )

    def transform_inductive(self, reference_features: np.ndarray, query_features: np.ndarray) -> HypergraphData:
        """Connect query patients only to training-reference patients.

        Hyperedge membership and weights are fixed from the reference patients.
        Query--query links are never created, so a held-out or external cohort
        cannot alter training-node degrees, normalisation, or edge weights.
        """

        if not self.fitted:
            raise RuntimeError("PatientHypergraphBuilder must be fitted before transform_inductive().")
        reference = _ensure_2d_float(reference_features)
        query = _ensure_2d_float(query_features)
        if reference.shape[1] != len(self.feature_specs) or query.shape[1] != len(self.feature_specs):
            raise ValueError("Feature width differs from the fitted hypergraph builder.")

        n_reference, n_query = len(reference), len(query)
        total = n_reference + n_query
        type_count = len(EDGE_TYPE_NAMES)
        accum = np.zeros((type_count, total, total), dtype=np.float32)
        degrees = np.zeros((type_count, total), dtype=np.float32)
        metadata: list[dict[str, object]] = []

        for spec, name, group in zip(self.feature_specs, self.feature_names, self.feature_groups):
            column = int(spec["column"])
            type_id = EDGE_TYPE_NAMES.index(group) if group in EDGE_TYPE_NAMES else EDGE_TYPE_NAMES.index("other")
            reference_groups = _groups_for_spec(reference[:, column], spec)
            query_groups = _groups_for_spec(query[:, column], spec)
            for group_id, reference_nodes in reference_groups.items():
                reference_nodes = np.asarray(reference_nodes, dtype=np.int64)
                if len(reference_nodes) < self.min_edge_size:
                    continue
                query_nodes = np.asarray(query_groups.get(group_id, []), dtype=np.int64) + n_reference
                weight = 1.0 / float(len(reference_nodes))
                accum[type_id][np.ix_(reference_nodes, reference_nodes)] += weight
                degrees[type_id, reference_nodes] += 1.0
                if len(query_nodes):
                    accum[type_id][np.ix_(reference_nodes, query_nodes)] += weight
                    accum[type_id][np.ix_(query_nodes, reference_nodes)] += weight
                    degrees[type_id, query_nodes] += 1.0
                metadata.append(
                    {
                        "name": f"{name}={group_id}",
                        "type": EDGE_TYPE_NAMES[type_id],
                        "reference_size": int(len(reference_nodes)),
                        "query_size": int(len(query_nodes)),
                    }
                )

        total_degrees = degrees.sum(axis=0)
        # Only reference nodes receive a fallback self-loop. A query self-loop
        # would constitute a query--query edge and would violate inductive use.
        isolated = np.where(total_degrees == 0)[0]
        isolated = isolated[isolated < n_reference]
        if len(isolated):
            other = EDGE_TYPE_NAMES.index("other")
            accum[other, isolated, isolated] = 1.0
            degrees[other, isolated] = 1.0
            for node in isolated:
                metadata.append({"name": f"self_loop_{node}", "type": "other", "reference_size": 1, "query_size": 0})

        for type_id in range(type_count):
            degree = degrees[type_id]
            scale = np.zeros_like(degree, dtype=np.float32)
            nonzero = degree > 0
            scale[nonzero] = 1.0 / np.sqrt(degree[nonzero])
            accum[type_id] = accum[type_id] * scale[:, None] * scale[None, :]

        return HypergraphData(
            propagation=accum.astype(np.float32),
            edge_count=len(metadata),
            edge_metadata=metadata,
        )


def validate_no_label_leakage(feature_names: Iterable[str], label_names: Iterable[str]) -> None:
    overlap = sorted(set(feature_names).intersection(set(label_names)))
    if overlap:
        raise ValueError(f"Label columns leaked into features: {overlap}")


def _normalize_category(value: object) -> str:
    if pd.isna(value):
        return "__MISSING__"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _ensure_2d_float(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D feature matrix.")
    return arr


def _groups_for_spec(values: np.ndarray, spec: dict[str, object]) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    if spec["kind"] == "binary":
        return {"active": np.where(values > 0.5)[0]}
    bins = np.digitize(values, np.asarray(spec["edges"], dtype=np.float32), right=False)
    return {f"bin_{bin_id}": np.where(bins == bin_id)[0] for bin_id in np.unique(bins)}
