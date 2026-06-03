# LogisticRegression iteration_2

## Experiment Note

Enhanced LogisticRegression with constant-feature removal, standardization, 5-fold CV, and repeated stratified splits.

## Data

- Samples: 351
- Features: 34
- Labels: {'b': 126, 'g': 225}
- Positive label for binary metrics: `g`

## Single Split Test Metrics

- test_accuracy: 0.9296
- test_precision: 0.9184
- test_recall: 0.9783
- test_f1: 0.9474
- test_roc_auc: 0.9243

## Outputs

- `single_split_metrics.csv`
- `classification_report.json`
- `confusion_matrix.csv`
- `confusion_matrix.png`
- `roc_curve.png`
- `cv_metrics.csv`
- `cv_metrics_summary.csv`
- `repeated_split_metrics.csv`
- `repeated_split_summary.csv`
