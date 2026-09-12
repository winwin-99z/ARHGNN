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

## Run locally under controlled access

```bash
python -m arhgnn.train --config configs/arhgnn.yaml
python -m arhgnn.evaluate --pred results_fixed/predictions_external.csv
python -m arhgnn.ablation --config configs/ablation.yaml
```


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
