# DecisionTree iteration_4

## Experiment Note

DecisionTree with fixed ccp_alpha from iteration_3 and GridSearchCV over max_depth and min_samples_leaf. Best params: {'max_depth': 8, 'min_samples_leaf': 1}.

## Data

- Samples: 351
- Features: 34
- Labels: {'b': 126, 'g': 225}
- Positive label for binary metrics: `g`

## Single Split Test Metrics

- test_accuracy: 0.9296
- test_precision: 0.9556
- test_recall: 0.9348
- test_f1: 0.9451
- test_roc_auc: 0.9374

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
