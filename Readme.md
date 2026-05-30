# 人工智能综合实训 II

本项目用于完成机器学习实训中的两个实验任务：

- **任务一：Ionosphere 传统机器学习分类**  
  使用逻辑回归、SVM、决策树、随机森林、AdaBoost 等模型完成二分类实验，并比较不同模型的效果。
- **任务二：Cats vs. Dogs 图像分类**  
  手动实现 AlexNet，对猫狗图片进行训练、验证和测试，并记录调参、改进与反思过程。

项目当前重点已经从“直接训练模型”前移到“先把数据质量和实验流程整理清楚”。原始数据、私有路径、大体量中间产物和审核结果默认不提交到仓库。

## 当前进度

- 已建立本地数据路径配置机制，代码不硬编码本机绝对路径。
- 已完成 Ionosphere 与 Cats vs. Dogs 的数据预分析。
- 已为 Cats vs. Dogs 建立异常图片检测流程：规则筛查、Isolation Forest、KMeans 距离与人工审核。
- 已制作异常图片静态审核包，方便分发给合作人审核。
- 已完成一轮重复图片候选检测与人工审核记录。
- 下一步应把异常、损坏、重复图片的处理结果合并为可复现的训练清单，再开始正式建模实验。

## 目录说明

```text
.
├── Readme.md                         # 项目入口说明
├── 流程.md                           # 实训总体流程
├── config/
│   ├── paths.example.json             # 可提交的路径配置模板
│   └── paths.local.json               # 本机私有路径配置，不提交
├── src/
│   ├── project_config.py              # 统一读取本地数据路径
│   ├── data_preanalysis.py            # 数据预分析脚本
│   ├── cats_anomaly_detection.py      # 猫狗异常图片候选检测
│   ├── cats_anomaly_review_gui.py     # 本地异常图片审核 Web 工具
│   ├── build_static_review_package.py # 生成异常图片静态审核包
│   └── build_duplicate_review_package.py # 生成重复图片审核包
├── docs/                              # 工作总结、分析记录等本地文档
├── exp/                               # 实验输出、图表和分析报告
└── data/                              # 清洗中间结果、审核结果等本地数据
```

其中 `data/`、`exp/`、`docs/` 以及 `config/paths.local.json` 默认是本地工作产物。仓库主要提交流程、代码和可公开的配置模板。

## 环境准备

按照项目约定，Python 开发默认使用 `dev` 环境：

```powershell
conda activate dev
```

如果后续需要安装新包，优先使用清华源。当前脚本主要依赖常见的数据分析与图像处理库，例如 `numpy`、`pandas`、`Pillow`、`scikit-learn`、`matplotlib`。

## 数据路径配置

不要在代码中写入本机绝对路径。首次运行前复制配置模板：

```powershell
Copy-Item config\paths.example.json config\paths.local.json
```

然后在 `config/paths.local.json` 中填写本机数据根目录。代码通过 `src/project_config.py` 读取配置，例如：

```python
from src.project_config import get_dataset_path

ionosphere_path = get_dataset_path("ionosphere")
cats_vs_dogs_path = get_dataset_path("cats_vs_dogs")
```

这样可以在本地快速读取数据，同时避免把私人路径提交到仓库。

## 常用命令

数据预分析：

```powershell
conda activate dev
python -m src.data_preanalysis --only all
```

猫狗异常图片检测：

```powershell
conda activate dev
python -m src.cats_anomaly_detection --top-fraction 0.03 --reuse-features
```

启动本地异常图片审核工具：

```powershell
conda activate dev
python -m src.cats_anomaly_review_gui --port 8765
```

生成可分发的异常图片静态审核包：

```powershell
conda activate dev
python -m src.build_static_review_package --zip-path path\to\cats_anomaly_review_package.zip
```

生成重复图片审核包：

```powershell
conda activate dev
python -m src.build_duplicate_review_package --zip-path path\to\cats_duplicate_review_package.zip
```

## 数据质量策略

本项目不直接删除原始图片。损坏图、异常图、重复图都应通过清单标记并在训练时排除或保留。

当前建议：

- **损坏图片**：记录并排除。
- **明确异常图片**：人工确认后排除，例如纯文字图、空白图、无猫狗主体、标签明显错误。
- **真实猫狗图片**：即使有白边、背景复杂、主体较小、光照较差、抠图背景等情况，也优先保留。
- **同标签重复图片**：每组保留一张，其余训练时排除，避免重复样本影响评估。
- **跨标签重复图片**：优先整组排除，避免标签冲突污染训练和测试。

后续训练应基于 manifest 文件，而不是直接扫描原始目录。

## 已知数据结论

Ionosphere：

- 样本数 351，特征数 34，二分类。
- 类别分布为 `b=126`、`g=225`。
- 无缺失值，有 1 条完全重复样本。
- 存在常量特征 `f2`。
- 样本量较小，后续评估应使用分层划分、交叉验证和多随机种子。

Cats vs. Dogs：

- 图片文件共 25000 张，其中 24998 张可正常读取。
- 已发现 2 张损坏图片。
- Cat 与 Dog 各 12500 张，类别整体平衡。
- 图片模式和尺寸不完全统一，训练前需要统一转为 RGB 并 resize/crop 到模型输入尺寸。
- 数据集中存在纯文字、低信息、重复、跨标签重复等质量问题，已经建立候选检测与人工审核流程。

## 下一步

1. 合并异常图片人工审核结果，形成最终异常排除清单。
2. 合并损坏图片、异常图片、重复图片处理结果，生成 `train_manifest.csv`、`val_manifest.csv`、`test_manifest.csv` 和 `excluded_images.csv`。
3. 开始 Ionosphere 传统机器学习建模，保存模型对比表、混淆矩阵、ROC 曲线和实验反思。
4. 实现 AlexNet 数据集读取、训练、验证和测试流程。
5. 在 AlexNet 上进行改进实验，例如数据增强、Batch Normalization、Dropout、学习率调整和全连接层规模调整。
6. 汇总实验结果，整理 Word 报告与答辩 PPT。

## 注意事项

- 异常图片相关输出统一保存在 `data/AlexNet/Anomaly/`。
- 所有清洗策略都应可复现：记录原因、保留清单，不直接改动原始数据。
- 如果需要删除或替换文件，默认先移动到 `.trash/`，除非已经明确确认可以安全删除。
