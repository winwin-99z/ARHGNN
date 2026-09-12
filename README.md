# ARHGNN

Reference implementation for the attention-residual hypergraph neural network
used for multi-label TCM pattern analysis in IgA nephropathy.

## Data protection and access

The clinical cohorts are private. This repository does not contain raw data,
patient identifiers, patient-level true labels, predicted probabilities,
thresholded predictions, or trained checkpoints. Access to the research data
and patient-level audit outputs is governed by the institutional data-access
process described in the manuscript.

Place authorised cohort files locally under `data/private/`; that directory and
the generated `results_fixed/` directory are ignored by version control. The
included `data/dataset_example.xlsx` is only a non-publication example for
format validation and must not be used to reproduce manuscript results.

## Configuration aligned with the manuscript

`configs/arhgnn.yaml` specifies the reported final architecture: two
attention-residual hypergraph layers, hidden dimension 32, BCE loss, label-set
SMOTE applied only after fixing the training split, a 0.5 threshold, up to 300
epochs, and early stopping with patience 50. The split configuration yields a
64/16/20 training/validation/held-out-test division of the primary cohort.

The implementation uses group-specific convolution kernels and concatenated
message/type-embedding attention with LeakyReLU, followed by
`LayerNorm(ReLU(Z) + residual)`, matching the manuscript equations.

## Run locally under controlled access

```bash
python -m arhgnn.train --config configs/arhgnn.yaml
python -m arhgnn.evaluate --pred results_fixed/predictions_external.csv
python -m arhgnn.ablation --config configs/ablation.yaml
```

The training command writes a controlled-access audit package in
`results_fixed/`, including pseudonymous record indices, cohort/split labels,
configuration snapshot, environment metadata, SHA-256 hashes, aggregate and
per-label metrics, calibration statistics, and a `results_manifest.json`.
The manifest verifies that element-wise accuracy plus Hamming loss equals one,
that support-weighted F1 is reproducible from the per-label values, and that
the confusion-matrix tally equals aggregate element-wise accuracy.

Do not publish the generated prediction CSV files. Deposit them only through an
approved controlled-access archive if required for editorial verification.

## Calibration and metric audit

Calibration uses equal-frequency bins: 10 bins for labels with at least 20
positive cases and 5 bins for sparse labels. The metric audit reports
element-wise accuracy, Hamming loss, subset accuracy, micro/macro metrics,
support-weighted F1, macro-AUC, macro-AUPRC, per-label TP/FP/FN/TN,
specificity, Brier score, ECE, and occupied-bin counts.

## Tests

```bash
python tests/run_tests.py
```

Tests use synthetic arrays and the included format example only; they do not
load private cohorts.
