# AdaBoost iteration_4

## Experiment Note

AdaBoost full GridSearchCV over n_estimators, learning_rate, and DecisionTree base-estimator depth, min_samples_leaf, criterion, class_weight. Best params: {'estimator__class_weight': None, 'estimator__criterion': 'entropy', 'estimator__max_depth': 2, 'estimator__min_samples_leaf': 2, 'learning_rate': 0.5, 'n_estimators': 50}.

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
- test_roc_auc: 0.9652

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
