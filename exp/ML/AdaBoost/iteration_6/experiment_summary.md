# AdaBoost iteration_6 — class_weight 对比实验

## Experiment Note

基于 iteration_4 GridSearchCV 选出的最优参数，对比基学习器（DecisionTreeClassifier）的两种 class_weight 配置：

- `None`（默认，不加权）
- `balanced`（按类别频率自动加权）

b:g = 126:225 ≈ 36:64，属于中等程度不平衡。

注意：iteration_4 的网格搜索已将 `estimator__class_weight` 纳入搜索空间（值为 `[None, "balanced"]`），搜索结果是 `None` 更优（以 macro_f1 为选择指标）。本次实验在最优参数基础上对两种配置做更详细的单次划分 + CV + 多次划分对比。

## 固定参数（来自 iteration_4 最优）

| 参数 | 值 |
|---|---|
| n_estimators | 50 |
| learning_rate | 0.5 |
| estimator（基学习器） | DecisionTreeClassifier |
| estimator__max_depth | 2 |
| estimator__min_samples_leaf | 2 |
| estimator__criterion | entropy |

## 单次划分对比（seed=42, test_size=0.2, stratify）

| Metric | None | balanced |
|---|---|---|
| test_accuracy | 0.9577 | 0.9577 |
| test_precision | 0.9574 | 0.9574 |
| test_recall | 0.9783 | 0.9783 |
| test_f1 | 0.9677 | 0.9677 |
| test_roc_auc | 0.9652 | 0.9857 |

## 5 折交叉验证（mean ± std）

| Metric | None | balanced |
|---|---|---|
| test_accuracy | **0.9458** ±0.0158 | 0.9173 ±0.0327 |
| test_precision | 0.9366 ±0.0141 | 0.9022 ±0.0314 |
| test_recall | 0.9822 ±0.0186 | 0.9778 ±0.0157 |
| test_f1 | **0.9588** ±0.0120 | 0.9384 ±0.0236 |
| test_roc_auc | 0.9612 ±0.0283 | 0.9611 ±0.0233 |

## 多次随机划分（5 seeds, mean ± std）

| Metric | None | balanced |
|---|---|---|
| test_accuracy | **0.9352** ±0.0161 | 0.9042 ±0.0336 |
| test_precision | **0.9263** ±0.0219 | 0.9081 ±0.0128 |
| test_recall | **0.9783** ±0.0000 | 0.9478 ±0.0451 |
| test_f1 | **0.9515** ±0.0115 | 0.9273 ±0.0268 |
| test_roc_auc | **0.9621** ±0.0137 | 0.9392 ±0.0291 |

## 结论

1. **class_weight='balanced' 对 AdaBoost 效果明确为负。** 多次划分下全部指标均下降，test_f1 从 0.9515 降至 0.9273，test_accuracy 从 0.9352 降至 0.9042。

2. 更关键的是 **稳定性明显变差**：test_accuracy 标准差从 0.0161 翻倍到 0.0336，recall 标准差从 0 飙升到 0.0451。说明 weighted 的 AdaBoost 对数据划分更敏感。

3. 原因分析：
   - b:g = 36:64 未达到需要 class_weight 的程度
   - AdaBoost 本身通过调整样本权重来处理难分样本，额外加 class_weight 造成了双重加权，过度关注少数类
   - iteration_4 GridSearchCV 选择 `class_weight=None` 的结论在此得到验证

4. **建议保留 class_weight=None（默认），与 iteration_4 网格搜索结果一致。**

## Outputs

- `None_metrics.json` / `balanced_metrics.json`
- `None_single_split_metrics.csv` / `balanced_single_split_metrics.csv`
- `None_cv_metrics.csv` / `balanced_cv_metrics.csv`
- `None_cv_metrics_summary.csv` / `balanced_cv_metrics_summary.csv`
- `None_repeated_split_metrics.csv` / `balanced_repeated_split_metrics.csv`
- `None_repeated_split_summary.csv` / `balanced_repeated_split_summary.csv`
- `class_weight_comparison.json`
