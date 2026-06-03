# DecisionTree iteration_3

## Experiment Note

DecisionTree with cost-complexity pruning. Only ccp_alpha is tuned from the pruning path by 5-fold CV macro F1. Best params: {'ccp_alpha': 0.00852480852480852}.

## Data

- Samples: 351
- Features: 34
- Labels: {'b': 126, 'g': 225}
- Positive label for binary metrics: `g`

## Single Split Test Metrics

- test_accuracy: 0.9155
- test_precision: 0.9348
- test_recall: 0.9348
- test_f1: 0.9348
- test_roc_auc: 0.8978

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
