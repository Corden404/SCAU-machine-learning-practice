# 人工智能综合实训 II

本仓库用于整理机器学习实训中的两个实验方向：

- **Ionosphere 传统机器学习分类**：围绕逻辑回归、SVM、决策树、随机森林、AdaBoost 与集成模型，比较单次划分、交叉验证和多随机种子重复划分下的表现。
- **Cats vs. Dogs AlexNet 图像分类**：手写并迭代 AlexNet 及其改进结构，结合数据清洗、固定 manifest 划分、训练曲线、混淆矩阵和错误样本分析说明模型改进依据。

## 目录结构

```text
.
├── Readme.md
├── config/
│   └── paths.example.json            # 可提交的本地路径配置模板
├── src/
│   ├── project_config.py             # 读取本地数据路径配置
│   ├── data_preanalysis.py           # Ionosphere 与 Cats vs. Dogs 数据预分析
│   ├── cats_anomaly_detection.py     # 猫狗异常图片候选检测
│   ├── cats_anomaly_review_gui.py    # 本地异常图片审核 Web 工具
│   ├── build_static_review_package.py
│   ├── build_duplicate_review_package.py
│   └── mimo_batch_review.py          # 可选的多模态辅助审核请求生成/调用
├── exp/
│   ├── ML/                           # 传统机器学习实验代码、记录和结果
│   └── AlexNet/                      # AlexNet 训练、模型、迭代与历史脚本
├── docs/                             # 本地过程文档，忽略
├── data/                             # 本地数据与清洗中间结果，忽略
└── presentation/                     # 汇报材料产物，忽略
```

## 实验说明

### Ionosphere

数据集特点：

- 样本数 351，特征数 34，二分类。
- 类别分布为 `b=126`、`g=225`。
- 无缺失值，有 1 条完全重复样本。
- 存在常量特征 `f2`。
- 样本量较小，因此实验重点是评估稳定性，而不是只看单次划分分数。

主要代码位于：

- `exp/ML/LogisticRegression/`
- `exp/ML/SVM/`
- `exp/ML/DecisionTree/`
- `exp/ML/RandomForest/`
- `exp/ML/AdaBoost/`
- `exp/ML/model_comparison/`
- `exp/ML/Ensemble/`

### Cats vs. Dogs / AlexNet

数据处理策略：

- 不直接删除原始图片。
- 损坏图、异常图、重复图通过清单记录。
- 训练应基于 manifest 文件，而不是直接扫描原始目录。
- 真实猫狗图片即使存在白边、复杂背景、主体较小、光照较差等情况，也优先保留。
- 跨标签重复或标签明显错误的图片应在训练清单中排除。

AlexNet 代码位于 `exp/AlexNet/`：

- `model.py`：模型结构定义。
- `common.py`：训练、manifest、评估、路径脱敏等公共逻辑。
- `iteration_*/run.py`：各轮迭代入口。
- `legacy/original_scripts/`：早期脚本归档。

