# SVM iteration_5

## Experiment Note

Exploratory full SVM search with StandardScaler, linear/RBF/poly kernels, C, gamma, class_weight, poly degree, and poly coef0. Best params: {'classifier__C': 3.0, 'classifier__class_weight': 'balanced', 'classifier__gamma': 0.03, 'classifier__kernel': 'rbf'}.

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
- test_roc_auc: 0.9765

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
