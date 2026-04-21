# toxicn

中文有害言论检测系统，包含两种检测方案：**BERT+FNN** 和 **基于形容词概念的MLP**。

## 项目结构

```
toxicnew/
├── configs/
│   ├── BERT_config.py          # BERT任务配置
│   └── MLP_config.py           # MLP任务配置
├── models/
│   ├── bert.py                 # BERTBaseline模型
│   └── mlp.py                  # MLP模型（门控机制）
├── scripts/
│   └── generate_adjective_c_r.py  # 形容词概念向量生成
├── utils/
│   ├── bert_pipeline.py        # BERT训练与测试流水线
│   ├── mlp_pipeline.py         # MLP训练与测试流水线
│   └── seed.py                 # 随机种子与确定性配置
├── data/
│   ├── raw/                    # 原始数据集与形容词词典
│   └── processed/              # 生成的概念向量
├── experiments_bert/           # BERT实验结果
└── experiments/                # MLP实验结果
```

## 任务一：BERT+FNN

### 模型架构

```
输入文本 → BertTokenizer编码 → BERT编码器 → [CLS]池化 → Dropout → 线性分类头 → 有毒/无毒
```

`BERTBaseline`：基于 bert-base-chinese 的二分类模型，全参数微调。结构为 BERT → [CLS] pooler_output → Dropout(0.1) → Linear(768→2)。

### 配置文件

`configs/BERT_config.py` — 主要参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `dataset_name` | 数据集名称 (TOXICN/COLD) | "TOXICN" |
| `bert_path` | bert-base-chinese 模型路径 | models/bert-base-chinese |
| `batch_size` | 批次大小 | 16 |
| `epochs` | 训练轮数 | 5 |
| `learning_rate` | 学习率 | 2e-5 |
| `warmup_ratio` | Warmup比例 | 0.1 |
| `weight_decay` | 权重衰减 | 0.01 |
| `max_seq_length` | 最大序列长度 | 128 |
| `dropout_rate` | Dropout比率 | 0.1 |
| `patience` | 早停耐心值 | 2 |

### 训练流程

1. 加载 `BERT_config.py` 默认配置，命令行参数可覆盖
2. 生成时间戳实验目录，保存配置到 `config.json`
3. 加载原始JSON数据，BertTokenizer编码，按9:1分层抽样划分训练/验证集
4. 训练：AdamW + linear warmup/decay + 梯度裁剪
5. 每个epoch记录验证集(Loss, F1, Precision, Recall)和测试集(Loss, F1，仅观察)
6. 基于验证集F1进行最佳模型选择和早停
7. 训练结束后绘制双子图曲线（上图Loss，下图Score）
8. all模式下自动评估最佳模型在测试集上的表现

### 使用方法

```bash
# 训练+测试
python utils/bert_pipeline.py --mode all

# 仅训练
python utils/bert_pipeline.py --mode train

# 仅测试（必须指定实验时间戳）
python utils/bert_pipeline.py --mode test --timestamp 20260421-120000

# 自定义超参数
python utils/bert_pipeline.py --mode all --dataset_name COLD --epochs 3 --batch_size 32 --patience 3
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式: all/train/test | all |
| `--timestamp` | 测试模式的实验时间戳 | - |
| `--dataset_name` | 数据集名称 (TOXICN/COLD) | TOXICN |
| `--seed` | 随机种子 | 1 |
| `--use_deterministic` | 启用确定性模式 | False |
| `--batch_size` | 批次大小 | 16 |
| `--epochs` | 训练轮数 | 5 |
| `--max_seq_length` | 最大序列长度 | 128 |
| `--patience` | 早停耐心值 | 2 |

参数优先级：训练模式 命令行 > BERT_config.py；测试模式强制使用实验目录的 config.json。

### 实验输出

```
experiments_bert/<timestamp>/
├── config.json              # 实验配置快照
├── best_model.pth           # 最佳模型权重
├── metrics.png              # 训练曲线图
└── test_results/            # 测试结果
    ├── metrics.json         # 评估指标
    └── classification_report.txt  # 分类报告
```

---

## 任务二：基于形容词概念的MLP

### 整体流程

```
原始文本 → 形容词概念向量生成(LLM) → MLP分类器 → 有毒/无毒判断
```

### 形容词概念向量生成

使用 Qwen2.5 系列大语言模型，对文本中每个形容词进行语义分析：

1. 构建提示模板：`"你是一个语义分析专家。请判断给定形容词是否准确描述文本，只回答'是'或'否'。"`
2. 批量推理：计算肯定词（"是"、"是的"、"对"等）的概率得分
3. 概念向量：将所有形容词的肯定概率组合成高维向量，形状为 `[形容词数量]`

采用公共前缀缓存 + 变动后缀批量推理优化，避免重复计算。

### MLP模型架构

```
输入: 形容词概念向量 [V]
    ↓
[门控层] Linear(V, V) + sigmoid → 学习特征重要性权重
    ↓
[Dropout]
    ↓
[隐藏层] V → 96 (ReLU + Dropout)
    ↓
[输出层] 96 → 2 (有毒/无毒)
```

门控机制通过 sigmoid 激活得到 0-1 范围的权重，元素级乘法实现自适应特征选择。

### 配置文件

`configs/MLP_config.py` — 主要参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `dataset_name` | 数据集名称 (TOXICN/COLD) | "TOXICN" |
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
| `patience` | 早停耐心值 | 20 |

### 使用方法

**第一步：生成形容词概念向量**

```bash
# 生成训练集的概念向量
python scripts/generate_adjective_c_r.py --mode train --dataset_name TOXICN --model_name Qwen2.5-1.5B-Instruct

# 生成测试集的概念向量
python scripts/generate_adjective_c_r.py --mode test --dataset_name TOXICN --model_name Qwen2.5-1.5B-Instruct
```

生成的概念向量文件保存在 `data/processed/` 目录下，命名格式为 `train_with_concepts(TOXICN)(Qwen2.5-1.5B-Instruct).json`。

> 生成概念向量需要GPU支持和较长时间。

**第二步：训练与测试**

```bash
# 训练+测试
python utils/mlp_pipeline.py --mode all

# 仅训练
python utils/mlp_pipeline.py --mode train

# 仅测试（必须指定实验时间戳）
python utils/mlp_pipeline.py --mode test --timestamp 20260415-085433

# 自定义超参数
python utils/mlp_pipeline.py --mode all --dataset_name COLD --epochs 100 --hidden_features 128 --dropout_rate 0.4
```

### MLP命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式: all/train/test | all |
| `--timestamp` | 测试模式的实验时间戳 | - |
| `--dataset_name` | 数据集名称 (TOXICN/COLD) | TOXICN |
| `--model_name` | LLM模型名称 | Qwen2.5-1.5B-Instruct |
| `--seed` | 随机种子 | 1 |
| `--use_deterministic` | 启用确定性模式 | False |
| `--batch_size` | 批次大小 | 16 |
| `--epochs` | 训练轮数 | 200 |
| `--max_lr` | 峰值学习率 | 1e-3 |
| `--pct_start` | Warmup比例 | 0.2 |
| `--div_factor` | 初始学习率除数 | 25.0 |
| `--final_div_factor` | 最终学习率除数 | 10000.0 |
| `--anneal_strategy` | 衰减策略 (cos/linear) | cos |
| `--dropout_rate` | Dropout比率 | 0.3 |
| `--hidden_features` | 隐藏层维度 | 96 |
| `--patience` | 早停耐心值 | 20 |

参数优先级：训练模式 命令行 > MLP_config.py；测试模式强制使用实验目录的 config.json。

### MLP实验输出

```
experiments/<timestamp>/
├── config.json              # 实验配置快照
├── best_model.pth           # 最佳模型权重
├── metrics.png              # 训练曲线图
└── test_results/
    ├── metrics.json         # 评估指标
    ├── classification_report.txt  # 分类报告
    └── predictions.json     # 逐条预测结果
```

---

## 公共说明

### 数据集

| 数据集 | 说明 |
|--------|------|
| TOXICN | 中文毒性言论数据集 |
| COLD | 中文冒犯性语言数据集 |

原始数据格式：
```json
[{"topic": "race", "content": "文本内容", "toxic": 1, "toxic_one_hot": [0, 1]}]
```

概念向量数据格式（MLP任务）：
```json
[{"content": "文本内容", "concept": [0.85, 0.12, 0.93, ..., 0.45]}]
```

### 评估指标

所有指标采用 **Macro Average**（宏平均）：
- **Precision**：预测为有毒的样本中真正有毒的比例
- **Recall**：真正有毒的样本中被正确预测的比例
- **F1 Score**：精确率和召回率的调和平均数

### 环境要求

```
Python >= 3.8
PyTorch >= 2.0
transformers >= 4.50
scikit-learn >= 1.0
matplotlib >= 3.5
pandas >= 1.5
bitsandbytes >= 0.40  # MLP任务：4bit量化加载LLM
```

安装依赖：
```bash
pip install -r requirements.txt
```

### 可复现性

两个任务均支持 `--use_deterministic` 参数启用确定性训练模式，配合 `--seed` 确保实验可复现。

## License

本项目仅供学术研究使用。
