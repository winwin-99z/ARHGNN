from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arhgnn.data import EDGE_TYPE_NAMES, FeaturePreprocessor, PatientHypergraphBuilder, load_dataset
from arhgnn.config import load_config
from arhgnn.metrics import (
    _labelset_stratified_bootstrap_indices,
    evaluate_multilabel,
    metric_invariants,
    read_prediction_csv,
    write_predictions,
)
from arhgnn.model import ARHGNN
from arhgnn.splits import split_train_val_test


class DataLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = load_dataset(ROOT / "data" / "dataset_example.xlsx")

    def test_example_dataset_is_format_only(self):
        self.assertEqual(self.example.patient_count, 49)
        self.assertEqual(self.example.labels.shape[1], 5)
        self.assertEqual(self.example.raw_features.shape[1], 41)

    def test_label_columns_are_not_features(self):
        overlap = set(self.example.feature_names).intersection(self.example.label_names)
        self.assertEqual(overlap, set())

    def test_splits_are_disjoint(self):
        splits = split_train_val_test(self.example.labels, seed=42)
        all_indices = set()
        for indices in splits.values():
            current = set(indices.tolist())
            self.assertFalse(all_indices.intersection(current))
            all_indices.update(current)
        self.assertEqual(len(all_indices), self.example.patient_count)

    def test_hypergraph_uses_encoded_features_only(self):
        splits = split_train_val_test(self.example.labels, seed=42)
        train_features = self.example.raw_features.iloc[splits["train_idx"]]
        preprocessor = FeaturePreprocessor().fit(train_features)
        for label in self.example.label_names:
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
            for label in self.example.label_names:
                self.assertNotIn(label, str(edge["name"]))

    def test_inductive_graph_has_no_query_to_query_edges(self):
        splits = split_train_val_test(self.example.labels, seed=42)
        preprocessor = FeaturePreprocessor().fit(self.example.raw_features.iloc[splits["train_idx"]])
        reference = preprocessor.transform(self.example.raw_features.iloc[splits["train_idx"]])
        query = preprocessor.transform(self.example.raw_features.iloc[splits["test_idx"]])
        builder = PatientHypergraphBuilder().fit(
            reference, preprocessor.encoded_feature_names, preprocessor.encoded_feature_groups
        )
        graph = builder.transform_inductive(reference, query)
        query_start = len(reference)
        self.assertTrue(np.allclose(graph.propagation[:, query_start:, query_start:], 0.0))


class MetricsTests(unittest.TestCase):
    def test_labelset_stratified_bootstrap_preserves_stratum_sizes(self):
        y_true = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 0]], dtype=np.float32)
        idx = _labelset_stratified_bootstrap_indices(y_true, np.random.default_rng(42))
        _, original_counts = np.unique(y_true, axis=0, return_counts=True)
        _, sampled_counts = np.unique(y_true[idx], axis=0, return_counts=True)
        np.testing.assert_array_equal(sampled_counts, original_counts)

    def test_probability_threshold_metrics(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
        y_prob = np.array([[0.60, 0.40], [0.40, 0.70], [0.51, 0.49], [0.20, 0.80]], dtype=np.float32)
        metrics = evaluate_multilabel(y_true, y_prob, threshold=0.5, bootstrap=0, label_names=["A", "B"])
        self.assertAlmostEqual(metrics["summary"]["elementwise_accuracy"]["mean"], 0.75)
        self.assertAlmostEqual(metrics["summary"]["hamming_loss"]["mean"], 0.25)
        self.assertAlmostEqual(metrics["summary"]["subset_accuracy"]["mean"], 0.5)
        self.assertEqual(metrics["per_label"][0]["support"], 2)
        self.assertEqual(metrics["per_label"][1]["support"], 2)
        self.assertIn("tn", metrics["per_label"][0])
        self.assertIn("expected_calibration_error", metrics["per_label"][0])
        invariants = metric_invariants(metrics)
        self.assertTrue(invariants["accuracy_plus_hamming_equals_one"])
        self.assertTrue(invariants["weighted_f1_reproducible_from_per_label_values"])
        self.assertTrue(invariants["figure8_tally_matches_elementwise_accuracy"])

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

    def test_group_specific_attention_model_forward_pass(self):
        model = ARHGNN(in_channels=3, hidden_channels=4, num_classes=2, edge_type_count=2, dropout=0.0)
        x = __import__("torch").randn(5, 3)
        propagation = __import__("torch").eye(5).repeat(2, 1, 1)
        logits = model(x, propagation)
        self.assertEqual(tuple(logits.shape), (5, 2))


class StaticChecks(unittest.TestCase):
    def test_final_configuration_matches_reported_architecture(self):
        config = load_config(ROOT / "configs" / "arhgnn.yaml")
        self.assertEqual(config["model"]["hidden_channels"], 32)
        self.assertEqual(config["training"]["epochs"], 300)
        self.assertEqual(config["evaluation"]["calibration"]["common_label_bins"], 10)
        self.assertEqual(config["evaluation"]["calibration"]["sparse_label_bins"], 5)

    def test_no_unreproducible_absolute_paths(self):
        forbidden = ["/opt" + "/data/private", "E" + ":/", "E" + ":\\"]
        for path in ROOT.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, msg=f"{token} found in {path}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
