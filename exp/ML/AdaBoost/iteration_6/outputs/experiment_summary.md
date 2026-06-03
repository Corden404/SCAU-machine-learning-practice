# AdaBoost iteration_6_class_weight_balanced

## Experiment Note

AdaBoost with best params from iteration_4, base estimator class_weight='balanced'. Comparison: None vs 'balanced'. (b:g ≈ 36:64, moderate imbalance). Note: iteration_4 GridSearchCV found class_weight=None performed best; this iteration re-runs both for a direct single-split + CV + repeated comparison.

## Data

- Samples: 351
- Features: 34
- Labels: {'b': 126, 'g': 225}
- Positive label for binary metrics: `g`

## Single Split Test Metrics

- test_accuracy: 0.9577
- test_precision: 0.9574
- test_recall: 0.9783
- test_f1: 0.9677
- test_roc_auc: 0.9857

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
