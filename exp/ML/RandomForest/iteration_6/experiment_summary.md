# RandomForest iteration_6 — class_weight 对比实验

## Experiment Note

基于 iteration_4 选出的最优参数，对比三种 class_weight 配置对随机森林的影响：

- `None`（默认，不加权）
- `balanced`（按类别频率自动加权）
- `balanced_subsample`（每次 bootstrap 采样时按类别频率加权）

b:g = 126:225 ≈ 36:64，属于中等程度不平衡。iteration_4 随机搜索未将 class_weight 纳入搜索空间，本次实验专门对比。

## 固定参数（来自 iteration_4 最优）

| 参数 | 值 |
|---|---|
| n_estimators | 100 |
| criterion | entropy |
| max_features | 0.5 |
| max_leaf_nodes | 16 |
| min_samples_leaf | 1 |
| min_samples_split | 2 |
| ccp_alpha | 0.0 |
| max_samples | 1.0 |
| bootstrap | True |

## 单次划分对比（seed=42, test_size=0.2, stratify）

| Metric | None | balanced | balanced_subsample |
|---|---|---|---|
| test_accuracy | 0.9437 | 0.9437 | 0.9437 |
| test_precision | 0.9565 | 0.9565 | 0.9565 |
| test_recall | 0.9565 | 0.9565 | 0.9565 |
| test_f1 | 0.9565 | 0.9565 | 0.9565 |
| test_roc_auc | 0.9765 | 0.9839 | 0.9813 |

## 5 折交叉验证（mean ± std）

| Metric | None | balanced | balanced_subsample |
|---|---|---|---|
| test_accuracy | 0.9487 ±0.0345 | 0.9373 ±0.0297 | 0.9345 ±0.0386 |
| test_precision | 0.9406 ±0.0276 | 0.9323 ±0.0272 | 0.9284 ±0.0311 |
| test_recall | 0.9822 ±0.0290 | 0.9733 ±0.0290 | 0.9733 ±0.0398 |
| test_f1 | 0.9609 ±0.0264 | 0.9522 ±0.0228 | 0.9500 ±0.0299 |
| test_roc_auc | 0.9778 ±0.0161 | 0.9744 ±0.0208 | 0.9757 ±0.0175 |

## 多次随机划分（5 seeds, mean ± std）

| Metric | None | balanced | balanced_subsample |
|---|---|---|---|
| test_accuracy | **0.9465** ±0.0209 | 0.9324 ±0.0184 | 0.9437 ±0.0223 |
| test_precision | 0.9384 ±0.0277 | 0.9296 ±0.0216 | 0.9381 ±0.0277 |
| test_recall | 0.9826 ±0.0097 | 0.9696 ±0.0194 | 0.9783 ±0.0154 |
| test_f1 | **0.9598** ±0.0152 | 0.9490 ±0.0138 | 0.9576 ±0.0164 |
| test_roc_auc | 0.9716 ±0.0101 | 0.9712 ±0.0099 | 0.9712 ±0.0109 |

## 结论

1. **class_weight='balanced' 和 'balanced_subsample' 对 RF 效果轻微为负。** 多次随机划分下 test_f1 从 0.9598 分别降至 0.9490 和 0.9576。

2. 单次划分下三个配置几乎一模一样，差异只在 ROC-AUC（balanced 略高）。但交叉验证和多次划分显示 None 更稳定。

3. 原因：b:g = 36:64 的不平衡程度不足以触发 class_weight 的正向效果。RF 的 bootstrap 采样本身就提供了一定的类别多样性；强制加权反而让模型过度补偿少数类。

4. **建议保留 class_weight=None（默认），不推荐对 Ionosphere 数据集使用 class_weight。**

## Outputs

- `None_metrics.json` / `balanced_metrics.json` / `balanced_subsample_metrics.json`
- `None_single_split_metrics.csv` / `balanced_single_split_metrics.csv` / `balanced_subsample_single_split_metrics.csv`
- `None_cv_metrics.csv` / `balanced_cv_metrics.csv` / `balanced_subsample_cv_metrics.csv`
- `None_cv_metrics_summary.csv` / `balanced_cv_metrics_summary.csv` / `balanced_subsample_cv_metrics_summary.csv`
- `None_repeated_split_metrics.csv` / `balanced_repeated_split_metrics.csv` / `balanced_subsample_repeated_split_metrics.csv`
- `None_repeated_split_summary.csv` / `balanced_repeated_split_summary.csv` / `balanced_subsample_repeated_split_summary.csv`
- `class_weight_comparison.json`
