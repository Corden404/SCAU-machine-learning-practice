# RandomForest iteration_6_class_weight_balanced_subsample

## Experiment Note

RandomForest with best params from iteration_4, class_weight='balanced_subsample'. Comparison: None vs 'balanced' vs 'balanced_subsample' (b:g ≈ 36:64, moderate imbalance).

## Data

- Samples: 351
- Features: 34
- Labels: {'b': 126, 'g': 225}
- Positive label for binary metrics: `g`

## Single Split Test Metrics

- test_accuracy: 0.9437
- test_precision: 0.9565
- test_recall: 0.9565
- test_f1: 0.9565
- test_roc_auc: 0.9813

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
