from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arhgnn.data import EDGE_TYPE_NAMES, FeaturePreprocessor, PatientHypergraphBuilder, load_dataset
from arhgnn.metrics import evaluate_multilabel, read_prediction_csv, write_predictions
from arhgnn.splits import split_train_val_test


class DataLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primary = load_dataset(ROOT / "data" / "IgAdata1.xlsx")
        cls.external = load_dataset(ROOT / "data" / "IgAdata2.xlsx", label_columns=cls.primary.label_names)

    def test_patient_counts_and_label_counts(self):
        self.assertEqual(self.primary.patient_count, 504)
        self.assertEqual(self.external.patient_count, 300)
        self.assertEqual(self.primary.labels.sum(axis=0).astype(int).tolist(), [264, 260, 363, 142, 25])
        self.assertEqual(self.external.labels.sum(axis=0).astype(int).tolist(), [163, 163, 214, 78, 14])

    def test_label_columns_are_not_features(self):
        overlap = set(self.primary.feature_names).intersection(self.primary.label_names)
        self.assertEqual(overlap, set())
        self.assertEqual(self.primary.raw_features.shape[1], 41)

    def test_splits_are_disjoint(self):
        splits = split_train_val_test(self.primary.labels, seed=42)
        all_indices = set()
        for indices in splits.values():
            current = set(indices.tolist())
            self.assertFalse(all_indices.intersection(current))
            all_indices.update(current)
        self.assertEqual(len(all_indices), self.primary.patient_count)

    def test_hypergraph_uses_encoded_features_only(self):
        splits = split_train_val_test(self.primary.labels, seed=42)
        train_features = self.primary.raw_features.iloc[splits["train_idx"]]
        preprocessor = FeaturePreprocessor().fit(train_features)
        for label in self.primary.label_names:
            self.assertFalse(any(label in name for name in preprocessor.encoded_feature_names))
        x_train = preprocessor.transform(train_features)
        builder = PatientHypergraphBuilder().fit(
            x_train,
            preprocessor.encoded_feature_names,
            preprocessor.encoded_feature_groups,
        )
        hg = builder.transform(x_train[:12])
        self.assertEqual(hg.propagation.shape, (len(EDGE_TYPE_NAMES), 12, 12))
        self.assertGreater(hg.edge_count, 0)
        for edge in hg.edge_metadata:
            for label in self.primary.label_names:
                self.assertNotIn(label, str(edge["name"]))


class MetricsTests(unittest.TestCase):
    def test_probability_threshold_metrics(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
        y_prob = np.array([[0.60, 0.40], [0.40, 0.70], [0.51, 0.49], [0.20, 0.80]], dtype=np.float32)
        metrics = evaluate_multilabel(y_true, y_prob, threshold=0.5, bootstrap=0, label_names=["A", "B"])
        self.assertAlmostEqual(metrics["summary"]["hamming_loss"]["mean"], 0.25)
        self.assertAlmostEqual(metrics["summary"]["subset_accuracy"]["mean"], 0.5)
        self.assertEqual(metrics["per_label"][0]["support"], 2)
        self.assertEqual(metrics["per_label"][1]["support"], 2)

    def test_prediction_roundtrip_without_eval(self):
        y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
        y_prob = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pred.csv"
            write_predictions(path, y_true, y_prob, ["A", "B"])
            loaded_true, loaded_prob, names = read_prediction_csv(path)
            np.testing.assert_array_equal(loaded_true, y_true)
            np.testing.assert_allclose(loaded_prob, y_prob)
            self.assertEqual(names, ["A", "B"])


class StaticChecks(unittest.TestCase):
    def test_no_unreproducible_absolute_paths(self):
        forbidden = ["/opt" + "/data/private", "E" + ":/", "E" + ":\\"]
        for path in ROOT.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, msg=f"{token} found in {path}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
