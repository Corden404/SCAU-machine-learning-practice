# LogisticRegression iteration_4

## Experiment Note

LogisticRegression with constant-feature removal, degree-2 PolynomialFeatures, StandardScaler, and GridSearchCV over L1, L2, and Elastic Net regularization. Best params: {'classifier__C': 3.0, 'classifier__l1_ratio': None, 'classifier__penalty': 'l2'}.

## Data

- Samples: 351
- Features: 34
- Labels: {'b': 126, 'g': 225}
- Positive label for binary metrics: `g`

## Single Split Test Metrics

- test_accuracy: 0.9577
- test_precision: 0.9388
- test_recall: 1.0000
- test_f1: 0.9684
- test_roc_auc: 0.9922

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
