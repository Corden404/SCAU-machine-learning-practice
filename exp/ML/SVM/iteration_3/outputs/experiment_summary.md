# SVM iteration_3

## Experiment Note

Greedy SVM iteration with StandardScaler and RBF SVC. Only C is tuned while gamma remains 'scale'. Best params: {'classifier__C': 3.0}.

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
- test_roc_auc: 0.9748

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
