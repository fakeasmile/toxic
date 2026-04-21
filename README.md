# toxicn

基于形容词概念向量的中文有害言论检测系统

## 项目简介

本项目是一个面向中文社交媒体的有害言论检测系统，采用**形容词概念向量 + MLP（多层感知机）**的创新架构。通过大型语言模型（LLM）提取文本中形容词的语义概念表示，结合门控机制的MLP分类器实现对有害言论的精准识别。

### 核心特性

- **形容词概念向量**：利用LLM分析文本中形容词的语义倾向，生成高维概念向量作为特征表示
- **门控机制MLP**：引入自适应门控单元，动态学习每个形容词特征的重要性权重
- **OneCycleLR调度**：支持先进的学习率调度策略，提升训练效率和模型性能
- **可复现性保证**：支持确定性训练模式，确保实验结果可复现
- **自动化流水线**：集成数据预处理、训练、测试的完整流程

## 分支管理策略

> **重要说明**：
> - **主分支（main/master）**：代码在不进行重大更新时保持不变
> - **其他分支**：均基于某一时刻从主分支 Fork 创建
> - **同步策略**：主分支不会主动同步代码到所有其他分支
> 
> 如需获取最新稳定版本，请使用主分支；如需实验性功能，请查看对应特性分支。

## 技术架构

### 1. 整体流程

```
原始文本 → 形容词概念向量生成 → MLP分类器 → 有毒/无毒判断
    ↓              ↓                    ↓
TOXICN/COLD   Qwen2.5 LLM         门控MLP网络
数据集        提取概念特征        (带注意力机制)
```

### 2. 形容词概念向量生成

使用Qwen2.5系列大语言模型，对文本中的每个形容词进行语义分析：

1. **构建提示模板**：`"你是一个语义分析专家。请判断给定形容词是否准确描述文本，只回答'是'或'否'。"`
2. **批量推理**：对形容词列表进行批量处理，计算肯定词（"是"、"是的"、"对"等）的概率得分
3. **概念向量**：将所有形容词的肯定概率组合成高维向量，形状为 `[形容词数量]`

### 3. MLP模型架构

```
输入: 形容词概念向量 [V]
    ↓
[门控层] ← 学习每个形容词特征的重要性权重 (sigmoid激活)
    ↓
[Dropout] ← 正则化，防止过拟合
    ↓
[隐藏层] V → 96 (ReLU激活 + Dropout)
    ↓
[输出层] 96 → 2 (有毒/无毒)
```

**门控机制**：
- 通过 `gate_layer = Linear(V, V)` 学习特征重要性
- 使用 sigmoid 激活得到 0-1 范围的权重
- 元素级乘法实现自适应特征选择

### 4. 形容词词典

项目包含245个精心筛选的形容词，涵盖：
- **有害性形容词**：abusive（辱骂性的）、hateful（仇恨的）、discriminatory（歧视性的）等
- **积极性形容词**：respectful（尊重的）、empathetic（共情的）、constructive（建设性的）等
- **中性形容词**：critical（批评的）、challenging（挑战性的）、controversial（有争议的）等

每个形容词配有中英文对照和详细定义说明。

## 环境要求

```
Python >= 3.8
PyTorch >= 2.0
transformers >= 4.50
modelscope >= 1.0
scikit-learn >= 1.0
matplotlib >= 3.5
pandas >= 1.5
bitsandbytes >= 0.40  # 用于4bit量化加载LLM
```

### 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 前置准备：生成形容词概念向量

在使用MLP模型之前，需要先使用LLM生成形容词概念向量文件：

```bash
# 生成训练集的概念向量
python scripts/generate_adjective_c_r.py --mode train --dataset_name TOXICN --model_name Qwen2.5-1.5B-Instruct

# 生成测试集的概念向量
python scripts/generate_adjective_c_r.py --mode test --dataset_name TOXICN --model_name Qwen2.5-1.5B-Instruct
```

生成的概念向量文件将保存在 `data/processed/` 目录下，命名格式为：
- `train_with_concepts(TOXICN)(Qwen2.5-1.5B-Instruct).json`
- `test_with_concepts(TOXICN)(Qwen2.5-1.5B-Instruct).json`

> **注意**：生成概念向量需要GPU支持和较长时间（取决于数据集大小），请耐心等待。

### 训练+测试一体化流程

使用统一的MLP流水线脚本，支持训练完成后自动测试：

```bash
# 默认配置（TOXICN数据集，Qwen2.5-1.5B-Instruct）
python utils/mlp_pipeline.py --mode all

# 指定COLD数据集
python utils/mlp_pipeline.py --mode all --dataset_name COLD

# 自定义超参数
python utils/mlp_pipeline.py --mode all \
    --dataset_name TOXICN \
    --batch_size 32 \
    --epochs 100 \
    --max_lr 5e-4 \
    --dropout_rate 0.4 \
    --hidden_features 128 \
    --seed 42

# 启用确定性模式（确保可复现）
python utils/mlp_pipeline.py --mode all --use_deterministic --seed 42
```

训练结果将保存在 `experiments/{timestamp}/` 目录下，包含：
- `config.json`：实验配置快照
- `best_model.pth`：最佳模型权重
- `metrics.png`：训练曲线可视化
- `test_results/`：测试结果目录
  - `metrics.json`：评估指标
  - `classification_report.txt`：详细分类报告
  - `predictions.json`：逐条预测结果

### 仅训练模式

```bash
python utils/mlp_pipeline.py --mode train --dataset_name TOXICN
```

### 仅测试模式

```bash
# 必须指定实验时间戳
python utils/mlp_pipeline.py --mode test --timestamp 20260415-085433
```

## 配置说明

### MLPConfig 主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `dataset_name` | 数据集名称 | "TOXICN" |
| `model_name` | LLM模型名称 | "Qwen2.5-1.5B-Instruct" |
| `batch_size` | 批次大小 | 16 |
| `epochs` | 训练轮数 | 200 |
| `max_lr` | 峰值学习率 | 1e-3 |
| `pct_start` | Warmup比例 | 0.2 |
| `div_factor` | 初始学习率除数 | 25.0 |
| `final_div_factor` | 最终学习率除数 | 10000.0 |
| `anneal_strategy` | 衰减策略 | "cos" |
| `dropout_rate` | Dropout比率 | 0.3 |
| `hidden_features` | 隐藏层维度 | 96 |
| `seed` | 随机种子 | 1 |
| `use_deterministic` | 启用确定性模式 | False |

### 命令行参数优先级

- **训练模式**：命令行参数 > MLP_config.py（命令行覆盖配置文件）
- **测试模式**：强制使用实验目录的 config.json（忽略命令行超参数）

这种设计确保了测试时使用与训练完全一致的配置，避免配置不一致导致的错误。

## 实验管理

所有实验结果按时间戳存储在 `experiments/` 目录下：

```
experiments/
├── 20260420-115958/
│   ├── config.json              # 实验配置快照
│   ├── best_model.pth           # 最佳模型权重
│   ├── metrics.png              # 训练曲线图
│   └── test_results/            # 测试结果目录
│       ├── metrics.json         # 评估指标
│       ├── classification_report.txt  # 分类报告
│       └── predictions.json     # 逐条预测结果
├── 20260420-123802/
│   ├── config.json
│   ├── best_model.pth
│   ├── metrics.png
│   └── test_results/
│       └── ...
└── ...
```

## 数据集

本项目支持以下中文有害言论数据集：

- **TOXICN**：中文毒性言论数据集，包含约2.9MB训练数据和730KB测试数据
- **COLD**：中文冒犯性语言数据集，包含约8.2MB训练数据和1.4MB测试数据

数据格式（原始数据）：
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

数据格式（概念向量）：
```json
[
  {
    "content": "文本内容",
    "concept": [0.85, 0.12, 0.93, ..., 0.45]  // 245维形容词概念向量
  }
]
```

## 模型评估

评估指标包括：
- **精确率（Precision）**：查准率，预测为有毒的样本中真正有毒的比例
- **召回率（Recall）**：查全率，真正有毒的样本中被正确预测的比例
- **F1 分数（F1 Score）**：精确率和召回率的调和平均数
- **准确率（Accuracy）**：所有样本中预测正确的比例

所有指标均采用 **Macro Average**（宏平均）计算，平等对待每个类别。

## 高级用法

### 自定义形容词词典

如需扩展形容词列表，可以编辑 `data/raw/adjective/toxic_adjectives.csv` 文件，添加新的形容词及其定义。重新生成概念向量后，MLP模型的输入维度会自动适配。

### 更换LLM模型

项目支持Qwen2.5系列不同规模的模型：
- `Qwen2.5-1.5B-Instruct`：轻量级，适合快速实验
- `Qwen2.5-3B-Instruct`：中等规模，平衡性能和速度
- `Qwen2.5-7B-Instruct`：大规模，可能获得更好的概念表示质量

```bash
python scripts/generate_adjective_c_r.py --mode train --dataset_name TOXICN --model_name Qwen2.5-3B-Instruct
python utils/mlp_pipeline.py --mode all --dataset_name TOXICN --model_name Qwen2.5-3B-Instruct
```

## 引用

如果您使用了本项目，请引用：

```
@software{toxicn,
  title = {toxicn: 基于形容词概念向量的中文有害言论检测系统},
  year = {2026}
}
```

## License

本项目仅供学术研究使用。
