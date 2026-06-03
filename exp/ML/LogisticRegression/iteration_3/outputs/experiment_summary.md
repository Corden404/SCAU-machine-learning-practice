# LogisticRegression iteration_3

## Experiment Note

LogisticRegression with constant-feature removal, StandardScaler, class_weight='balanced', and C selected by 5-fold CV macro F1 from [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]. Best C=3.0.

## Data

- Samples: 351
- Features: 34
- Labels: {'b': 126, 'g': 225}
- Positive label for binary metrics: `g`

## Single Split Test Metrics

- test_accuracy: 0.8732
- test_precision: 0.8936
- test_recall: 0.9130
- test_f1: 0.9032
- test_roc_auc: 0.8983

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
