# LogisticRegression_interaction iteration_5

## Experiment Note

LogisticRegression interaction_only (degree=2, no squared terms). GridSearchCV over L1/L2/ElasticNet. Best params: {'clf__C': 0.01, 'clf__l1_ratio': None, 'clf__penalty': 'l2'}. Polynomial features: 561.

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
- test_roc_auc: 0.9800

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
