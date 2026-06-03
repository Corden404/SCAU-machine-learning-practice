# AdaBoost记录

## 1. 当前迭代概况

目前 AdaBoost 已完成五次有效迭代。

1. `iteration_1` 使用默认参数 AdaBoost，作为单次划分 baseline。
2. `iteration_2` 将 `learning_rate` 降为 `0.5`，并增加 5 折交叉验证和多随机种子重复划分，用于观察稳定性。
3. `iteration_3` 恢复默认参数，并补齐 5 折交叉验证和多随机种子重复划分，用于公平比较默认参数和小学习率方案。
4. `iteration_4` 一次性搜索 AdaBoost 主要超参数和弱学习器决策树参数，观察全参数维度调参是否进一步提升。
5. `iteration_5` 以 `iteration_3` 和 `iteration_4` 的模型为基础，做分类阈值分析，发现 iter4 从阈值调整中获益。

## 2. iteration_1 默认参数结果

默认参数：

```text
AdaBoostClassifier(random_state=seed)
```

单次划分结果：

| 指标 | 训练集 | 测试集 |
| --- | ---: | ---: |
| accuracy | 0.9821 | 0.9577 |
| precision(g) | 0.9728 | 0.9388 |
| recall(g) | 1.0000 | 1.0000 |
| F1(g) | 0.9862 | 0.9684 |
| ROC-AUC | 0.9997 | 0.9791 |

测试集分类结果：

| 类别 | precision | recall | F1 |
| --- | ---: | ---: | ---: |
| b | 1.0000 | 0.8800 | 0.9362 |
| g | 0.9388 | 1.0000 | 0.9684 |
| macro avg | 0.9694 | 0.9400 | 0.9523 |

混淆矩阵：

```text
actual_b: pred_b=22, pred_g=3
actual_g: pred_b=0,  pred_g=46
```

这一轮说明默认 AdaBoost 在当前固定划分上表现很强，尤其 `g` 类完全召回，整体 accuracy 和 ROC-AUC 都较高。但训练集 ROC-AUC 接近 1，说明模型已经把训练集排序得几乎完全正确，存在一定过拟合风险。并且 `b` 类 recall 只有 `0.8800`，错误集中在把少数类 `b` 判成 `g`。

这一轮最大的问题是没有交叉验证和多随机种子重复划分，因此只能说明这个固定划分上表现好，不能证明它稳定。

## 3. iteration_2 降低 learning_rate 结果

参数：

```text
AdaBoostClassifier(learning_rate=0.5, random_state=seed)
```

单次划分结果：

| 指标 | 训练集 | 测试集 |
| --- | ---: | ---: |
| accuracy | 0.9536 | 0.9437 |
| precision(g) | 0.9368 | 0.9375 |
| recall(g) | 0.9944 | 0.9783 |
| F1(g) | 0.9648 | 0.9574 |
| ROC-AUC | 0.9895 | 0.9639 |

测试集分类结果：

| 类别 | precision | recall | F1 |
| --- | ---: | ---: | ---: |
| b | 0.9565 | 0.8800 | 0.9167 |
| g | 0.9375 | 0.9783 | 0.9574 |
| macro avg | 0.9470 | 0.9291 | 0.9371 |

混淆矩阵：

```text
actual_b: pred_b=22, pred_g=3
actual_g: pred_b=1,  pred_g=45
```

稳定性评估：

| 评估方式 | accuracy | F1(g) | ROC-AUC |
| --- | ---: | ---: | ---: |
| 5 折 CV 均值 | 0.9259 | 0.9443 | 0.9589 |
| 5 折 CV 标准差 | 0.0370 | 0.0285 | 0.0213 |
| 重复划分均值 | 0.9239 | 0.9437 | 0.9557 |
| 重复划分标准差 | 0.0161 | 0.0113 | 0.0210 |

这一轮说明降低 `learning_rate` 起到了正则化作用：训练集 accuracy 从 `0.9821` 降到 `0.9536`，训练集 ROC-AUC 从 `0.9997` 降到 `0.9895`，模型没有上一轮那么贴合训练集。

但是从单次测试集看，`learning_rate=0.5` 没有带来提升，accuracy、F1(g)、ROC-AUC 都低于默认参数；`b` 类 recall 仍然是 `0.8800`，并没有缓解少数类被判成 `g` 的问题。

## 4. 前两次迭代说明什么

1. 默认 AdaBoost 已经是一个较强 baseline，单次划分效果明显优于默认单棵决策树，也接近前面较好的模型表现。
2. 单纯降低 `learning_rate` 不是有效改进，至少在当前固定划分上会让 accuracy、F1 和 ROC-AUC 同时下降。
3. `learning_rate=0.5` 不能直接判定为失败，因为它和 `n_estimators` 明显相互影响。学习率降低后，通常需要更多弱分类器轮数来补足表达能力。
4. 两轮共同暴露的主要问题是 `b` 类 recall 偏低。模型更倾向于保护多数类 `g` 的召回，少数类 `b` 仍有一部分被错判成 `g`。
5. 当前还不能公平比较默认参数和 `learning_rate=0.5` 的泛化稳定性，因为 `iteration_1` 没有跑 CV 和重复划分。

## 5. iteration_3 默认参数稳定性评估结果

参数：

```text
AdaBoostClassifier(random_state=seed)
```

单次划分结果与 `iteration_1` 完全一致：

| 指标 | 训练集 | 测试集 |
| --- | ---: | ---: |
| accuracy | 0.9821 | 0.9577 |
| precision(g) | 0.9728 | 0.9388 |
| recall(g) | 1.0000 | 1.0000 |
| F1(g) | 0.9862 | 0.9684 |
| ROC-AUC | 0.9997 | 0.9791 |

新增稳定性评估：

| 评估方式 | accuracy | F1(g) | ROC-AUC |
| --- | ---: | ---: | ---: |
| 5 折 CV 均值 | 0.9401 | 0.9545 | 0.9608 |
| 5 折 CV 标准差 | 0.0468 | 0.0358 | 0.0282 |
| 重复划分均值 | 0.9296 | 0.9473 | 0.9631 |
| 重复划分标准差 | 0.0263 | 0.0201 | 0.0221 |

与 `iteration_2` 对比：

| 指标 | iteration_2 learning_rate=0.5 | iteration_3 默认参数 |
| --- | ---: | ---: |
| CV accuracy | 0.9259 | 0.9401 |
| CV F1(g) | 0.9443 | 0.9545 |
| CV ROC-AUC | 0.9589 | 0.9608 |
| repeated accuracy | 0.9239 | 0.9296 |
| repeated F1(g) | 0.9437 | 0.9473 |
| repeated ROC-AUC | 0.9557 | 0.9631 |

这一轮说明默认 AdaBoost 的单次高分不是完全偶然。恢复默认参数后，CV 和重复划分均值都比 `learning_rate=0.5` 更好，尤其 CV accuracy 和 F1(g) 有明确提升。

但默认参数的训练集表现也更接近满分，训练集 ROC-AUC 仍接近 `1.0000`。因此它不是“无过拟合”的模型，而是在当前数据上用更强拟合换来了更好的泛化均值。

## 6. iteration_4 全参数维度搜索结果

本轮同时搜索：

1. AdaBoost 自身参数：`n_estimators`、`learning_rate`。
2. 弱学习器决策树参数：`max_depth`、`min_samples_leaf`、`criterion`、`class_weight`。

候选组合共 `576` 组。主选择指标为 5 折 CV 的 `macro F1`，同时观察 `b` 类 recall、`g` 类 F1、accuracy、ROC-AUC。`algorithm` 没有搜索，因为当前 sklearn 版本中该参数已废弃。

最优参数：

```text
estimator__class_weight = None
estimator__criterion = entropy
estimator__max_depth = 2
estimator__min_samples_leaf = 2
learning_rate = 0.5
n_estimators = 50
```

网格搜索最佳 `macro F1` 为 `0.9399`。

单次划分结果：

| 指标 | 训练集 | 测试集 |
| --- | ---: | ---: |
| accuracy | 1.0000 | 0.9577 |
| precision(g) | 1.0000 | 0.9574 |
| recall(g) | 1.0000 | 0.9783 |
| F1(g) | 1.0000 | 0.9677 |
| ROC-AUC | 1.0000 | 0.9652 |

测试集分类结果：

| 类别 | precision | recall | F1 |
| --- | ---: | ---: | ---: |
| b | 0.9583 | 0.9200 | 0.9388 |
| g | 0.9574 | 0.9783 | 0.9677 |
| macro avg | 0.9579 | 0.9491 | 0.9533 |

混淆矩阵：

```text
actual_b: pred_b=23, pred_g=2
actual_g: pred_b=1,  pred_g=45
```

稳定性评估：

| 评估方式 | accuracy | F1(g) | ROC-AUC |
| --- | ---: | ---: | ---: |
| 5 折 CV 均值 | 0.9458 | 0.9588 | 0.9612 |
| 5 折 CV 标准差 | 0.0158 | 0.0120 | 0.0283 |
| 重复划分均值 | 0.9352 | 0.9515 | 0.9621 |
| 重复划分标准差 | 0.0161 | 0.0115 | 0.0137 |

与 `iteration_3` 对比：

| 指标 | iteration_3 默认参数 | iteration_4 全参数搜索 |
| --- | ---: | ---: |
| 单次 accuracy | 0.9577 | 0.9577 |
| 单次 F1(g) | 0.9684 | 0.9677 |
| 单次 ROC-AUC | 0.9791 | 0.9652 |
| 单次 b recall | 0.8800 | 0.9200 |
| CV accuracy | 0.9401 | 0.9458 |
| CV F1(g) | 0.9545 | 0.9588 |
| CV ROC-AUC | 0.9608 | 0.9612 |
| repeated accuracy | 0.9296 | 0.9352 |
| repeated F1(g) | 0.9473 | 0.9515 |
| repeated ROC-AUC | 0.9631 | 0.9621 |

这一轮说明全参数维度搜索有小幅收益。相比默认参数，CV 和重复划分下的 accuracy、F1(g) 均提升，单次测试集的 `b` 类 recall 也从 `0.8800` 提高到 `0.9200`。

但它也明显更容易过拟合：训练集 accuracy、F1、ROC-AUC 都达到 `1.0000`。因此 `iteration_4` 可以作为当前 AdaBoost 的最佳分类指标版本，但不能说它比 `iteration_3` 更稳健。

## 7. 后续优化思路

建议按贪心迭代继续，不一次性把所有参数都打开。

### 后续建议

AdaBoost 已经通过全参数维度搜索拿到小幅提升。继续优化时，不建议再盲目扩大网格，而应该围绕 `iteration_4` 的最优组合做局部分析。

### 后续可能方向

1. 若追求分类指标，可以保留 `iteration_4` 作为 AdaBoost 当前最佳版本。
2. 若担心过拟合，可以回退到 `iteration_3`，或围绕 `iteration_4` 尝试更保守的局部参数，例如固定 `max_depth=2`，减少 `n_estimators` 或降低 `learning_rate`。
3. 若重点提升少数类 `b`，可以做阈值调整，观察 `b` recall 和 `g` F1 的权衡。
4. 暂不优先做标准化和多项式升维：AdaBoost 的默认弱学习器是树模型，对特征尺度不敏感；多项式升维会显著增加特征数量，可能让 Boosting 更容易追逐噪声。

## 8. 当前结论

当前 AdaBoost 最好分类指标版本是 `iteration_4`：

```text
AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        criterion="entropy",
        max_depth=2,
        min_samples_leaf=2,
        class_weight=None,
    ),
    learning_rate=0.5,
    n_estimators=50,
)
```

它比默认参数有小幅泛化收益，并改善了单次测试集的 `b` 类 recall。但它训练集已达到满分，报告中应明确说明存在更强过拟合风险。

## 9. iteration_5 阈值分析

本轮不再调整弱学习器结构，改为对 `iteration_3` 默认参数和 `iteration_4` 全参数搜索模型做分类阈值分析。

阈值范围 0.10~0.90，步长 0.02，5 个随机种子 repeated split 评估。

### 9.1 AdaBoost iter4 阈值分析

| 阈值策略 | 阈值 | accuracy | macro F1 | F1(g) | F1(b) | recall(b) | recall(g) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 默认 | 0.50 | 0.9352 | 0.9270 | 0.9515 | 0.9024 | 0.8560 | 0.9783 |
| 最优 macro F1 | 0.48 | 0.9408 | **0.9327** | 0.9561 | 0.9093 | 0.8480 | 0.9913 |
| 最优 accuracy | 0.48 | 0.9408 | 0.9327 | 0.9561 | 0.9093 | 0.8480 | 0.9913 |
| 最优 recall(b) | 0.72 | 0.5775 | 0.5696 | 0.5135 | 0.6257 | 1.0000 | 0.3478 |

### 9.2 AdaBoost iter3 阈值分析

| 阈值策略 | 阈值 | accuracy | macro F1 | F1(g) | F1(b) | recall(b) | recall(g) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 默认 | 0.50 | 0.9296 | 0.9206 | 0.9473 | 0.8940 | 0.8400 | 0.9783 |
| 最优 macro F1 | 0.50 | 0.9296 | 0.9206 | 0.9473 | 0.8940 | 0.8400 | 0.9783 |

### 9.3 阈值分析结论

1. **iter4 从阈值调整中获益，iter3 不获益。** iter4 将阈值从 0.5 微调至 0.48，macro F1 从 0.9270 升至 0.9327（+0.0057），accuracy 从 0.9352 升至 0.9408（+0.0056）。iter3 的默认阈值已经最优。
2. **调整方向是略微降低阈值。** 阈值 0.48 意味着更倾向于判为 g 类。AdaBoost 对 g 类的概率估计偏保守，需要略微降低门槛。
3. **与 RF 阈值分析的关键差异。** RF 的默认 0.5 已经最优，阈值调整几乎无收益；AdaBoost iter4 却有约 0.006 的改善。说明 AdaBoost 的概率校准略逊于 RF。
4. **b recall 的提升仍然代价巨大。** 阈值 0.72 可让 b recall 达到 1.0，但 accuracy 跌至 0.5775。
5. **推荐将 iter4 + 阈值 0.48 作为 AdaBoost 新最佳版本。** 在模型对比中，这使 AdaBoost macro F1 从 0.9270 提升到 0.9327，几乎追平 LR（0.9331）。

## 10. 后续可能方向

1. 以 `iteration_4 + threshold=0.48` 作为 AdaBoost 最终推荐版本，更新到模型对比中。
2. 不建议继续对 AdaBoost 调参或调阈值。iter4 训练集已满分，继续增加复杂度只会加重过拟合。
3. 后续工作应进入最终模型对比阶段。
