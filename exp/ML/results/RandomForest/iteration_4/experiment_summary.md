# RandomForest iteration_4

## Experiment Note

RandomForest local RandomizedSearchCV around iteration_2/3, followed by repeated-split reranking of the top CV candidates. Selected params: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_samples': 1.0, 'max_leaf_nodes': 16, 'max_features': 0.5, 'criterion': 'entropy', 'ccp_alpha': 0.0}.

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
