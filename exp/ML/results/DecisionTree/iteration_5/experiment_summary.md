# DecisionTree iteration_5

## Experiment Note

DecisionTree criterion comparison (gini vs entropy vs log_loss) with fixed ccp_alpha=0.00852480852480852 from iteration_3. Best criterion: {'criterion': 'entropy'}.

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
- test_roc_auc: 0.9074

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
