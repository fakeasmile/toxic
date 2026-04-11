# toxicn

基于 BERT + FNN 的中文有害言论检测与识别系统

## 项目简介

本项目是一个面向中文社交媒体的有害言论检测系统，采用 **BERT + FNN（前馈神经网络）** 的架构，结合**毒性词典**和**攻击立场词典**进行多特征融合，实现对有害言论的精准识别。

### 核心特性

- **多特征融合**：在 BERT 词嵌入层融合毒性词嵌入（toxic embeddings）和攻击立场嵌入（stance embeddings）
- **学习率动态调度**：支持 OneCycleLR 等学习率调度策略
- **早停机制**：防止过拟合，提升模型泛化能力
- **可复现性保证**：支持确定性训练模式
- **实验管理**：自动化的实验目录管理和结果可视化

> **注意**：本项目包含两个独立任务
> 1. **BERT+FNN**：基于 BERT 嵌入 + 全连接网络的有害言论检测（使用 TOXICN/COLD 数据集）
> 2. **MLP**：基于形容词概念向量的 MLP 分类器（使用 LLM 生成的概念向量）
> 
> 两个任务目前独立运行，后续将整合为统一框架。超参数搜索模块当前不使用。

## 项目结构

```
toxicn/
├── configs/                    # 配置文件
│   ├── base_config.py          # 基础配置（BERT+FNN）
│   ├── MLP_config.py           # MLP模型配置
│   └── hyperopt_config.py      # 超参数搜索配置
├── data/                       # 数据目录
│   ├── raw/                    # 原始数据
│   │   ├── COLD/               # COLD数据集
│   │   ├── TOXICN/             # TOXICN数据集
│   │   ├── adjective/          # 形容词词典
│   │   ├── lexicon/            # 毒性词典（general, LGBT, racism, region, sexism）
│   │   └── attack_stance.json  # 攻击立场词典
│   └── processed/              # 处理后数据（概念向量）
├── models/                     # 模型定义
│   ├── bert.py                 # 改进版BERT（支持自定义嵌入层）
│   ├── fc.py                   # 全连接分类器
│   └── mlp.py                  # MLP分类器（带门控机制）
├── utils/                      # 工具函数
│   ├── data_preprocess.py      # 数据预处理和Dataset
│   ├── train.py                # BERT+FNN训练流程
│   ├── test.py                 # 模型测试与评估
│   ├── hyperopt.py             # 超参数搜索主逻辑
│   ├── hyperopt_train.py       # 超参数搜索训练流程
│   ├── mlp_train.py            # MLP训练流程
│   ├── mlp_test.py             # MLP测试流程
│   └── seed.py                 # 随机种子设置
├── scripts/                    # 分析脚本
│   ├── analyze_dataset.py      # 数据集统计分析
│   └── generate_adjective_c_r.py # 生成形容词概念向量
└── experiments/                # 实验结果保存目录
```

## 技术架构

### 1. 模型架构

```
输入文本
    ↓
[CustomEmbeddings] ← toxic_ids, stance_ids
    ↓
[BERT Encoder] (冻结底层，训练顶层)
    ↓
[Pooler Output] (768维)
    ↓
[FC Classifier] (768 → 256 → 2)
    ↓
输出（有毒/无毒）
```

### 2. 自定义嵌入层

在标准 BERT 的三种嵌入（Token + Segment + Position）基础上，额外添加：

- **毒性嵌入（Toxic Embeddings）**：标记文本中的脏词位置，支持 6 种毒性类别
- **攻击立场嵌入（Stance Embeddings）**：标记具有攻击立场的词汇

### 3. 词典资源

- **毒性词典**：包含 general、LGBT、racism、region、sexism 五类脏词
- **攻击立场词典**：包含 region、race、gender、LGBT、political、other 六类攻击立场模式

## 环境要求

```
Python >= 3.8
PyTorch >= 2.0
transformers >= 4.50
```

### 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 配置数据集

修改 `configs/base_config.py` 中的数据集路径：

```python
self.dataset_name = "TOXICN"  # 或 "COLD"
```

### 2. 训练 BERT+FNN 模型

```python
from utils.train import init, final_train

# 初始化配置
base_config = init()

# 开始训练
final_train(base_config)
```

训练结果将保存在 `experiments/{timestamp}/` 目录下，包含：
- `best_model.pth`：最佳模型权重
- `config.json`：训练配置
- `metrics.png`：训练指标可视化

### 3. 测试 BERT+FNN 模型

```python
from utils.test import test
from pathlib import Path

# 指定实验目录
experiment_dir = Path("experiments/20260411-161544")
test(experiment_dir)
```

## 配置说明

### BaseConfig 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `dataset_name` | 数据集名称 | "TOXICN" |
| `bert_path` | BERT 模型路径 | "models/bert-base-chinese" |
| `batch_size` | 批次大小 | 64 |
| `epochs` | 训练轮数 | 8 |
| `lr` | 学习率 | 1e-5 |
| `max_len` | 最大序列长度 | 80 |
| `dropout_rate` | Dropout 比率 | 0.5 |
| `freeze_bert_layers` | 冻结 BERT 层数 | 0 |
| `num_toxic_types` | 毒性类别数 | 6 |
| `use_deterministic` | 启用确定性模式 | True |

## 实验管理

所有实验结果按时间戳存储在 `experiments/` 目录下：

```
experiments/
├── 20260410-164833/
│   ├── best_model.pth      # 模型权重
│   ├── config.json         # 实验配置
│   └── metrics.png         # 训练曲线
├── 20260411-161544/
│   ├── best_model.pth
│   ├── config.json
│   ├── metrics.png
│   └── test_results/       # 测试结果
│       └── test_metrics.json
└── ...
```

## 数据集

本项目支持以下中文有害言论数据集：

- **TOXICN**：中文毒性言论数据集
- **COLD**：中文冒犯性语言数据集

数据格式：
```json
[
  {
    "topic": "race",
    "content": "文本内容",
    "toxic": 1,
    "toxic_one_hot": [0, 1]
  }
]
```

## 模型评估

评估指标包括：
- **精确率（Precision）**
- **召回率（Recall）**
- **F1 分数（F1 Score）**
- **准确率（Accuracy）**

## MLP 任务（独立运行）

基于形容词概念向量的 MLP 分类器（带门控机制），与 BERT+FNN 任务独立：

### 训练 MLP 模型

```python
from utils.mlp_train import init, load_data, train

mlp_config = init()
train_data = load_data(mlp_config, "train_with_concepts(TOXICN)(Qwen2.5-1.5B-Instruct).json")
test_data = load_data(mlp_config, "test_with_concepts(TOXICN)(Qwen2.5-1.5B-Instruct).json")
train(mlp_config, train_data, test_data)
```

### 测试 MLP 模型

```python
from utils.mlp_test import evaluate_best_model

evaluate_best_model("20260410-164833")
```

### 数据分析

```python
from scripts.analyze_dataset import analyze_toxic_dataset
from utils.data_preprocess import ToxicDataset

# 分析数据集统计信息
analyze_toxic_dataset(dataset, "train")
```

## 引用

如果您使用了本项目，请引用：

```
@software{toxicn,
  title = {toxicn: BERT+FNN中文有害言论检测系统},
  year = {2026}
}
```

## License

本项目仅供学术研究使用。
