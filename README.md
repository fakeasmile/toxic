# toxicn

中文有害言论检测系统，基于**形容词概念的MLP**实现。

## 项目结构

```
toxicnew/
├── configs/
│   └── MLP_config.py           # MLP任务配置
├── models/
│   └── mlp.py                  # MLP模型（门控机制）
├── scripts/
│   ├── generate_adjective_c_r_vllm.py  # 形容词概念向量生成 (vLLM)
│   ├── inspect_prompt_template_vllm.py  # 调试提示词
│   └── inspect_verbalizer_coverage_vllm.py  # 调试verbalizer
├── utils/
│   ├── mlp_pipeline.py        # MLP训练与测试流水线
│   └── seed.py                # 随机种子与确定性配置
├── data/
│   ├── raw/                  # 原始数据集与形容词词典
│   └── processed/              # 生成的概念向量
└── experiments/                # MLP实验结果
```

## 整体流程

```
原始文本 → 形容词概念向量生成(LLM) → MLP分类器 → 有毒/无毒判断
```

## 形容词概念向量生成

使用 Qwen2.5 系列大语言模型，对文本中每个形容词进行语义分析：

1. 构建Likert式提示模板，评估形容词与文本的相关程度
2. 使用vLLM批量推理，支持AWQ/FP8量化，通过首token logprobs计算概率
3. 计算177个形容词的得分，组成高维概念向量
4. 支持GPU显存可配置

## MLP模型架构

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

## 配置文件

`configs/MLP_config.py` — 主要参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `dataset_name` | 数据集名称 (TOXICN/COLD) | "TOXICN" |
| `model_name` | LLM模型名称 | "Qwen2.5-1.5B-Instruct" |
| `template` | 提示词模板类型 | "likert" |
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

## 使用方法

**第一步：生成形容词概念向量**

```bash
# 生成训练集的概念向量（无量化）
python scripts/generate_adjective_c_r_vllm.py --mode train --dataset_name TOXICN --model_name Qwen2.5-1.5B-Instruct --template likert

# 生成测试集的概念向量（AWQ量化）
python scripts/generate_adjective_c_r_vllm.py --mode test --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ --template likert --quantization awq

# 自定义GPU显存占用
python scripts/generate_adjective_c_r_vllm.py --mode train --dataset_name TOXICN --model_name Qwen2.5-1.5B-Instruct --template likert --gpu_memory_utilization 0.8
```

生成的概念向量文件保存在 `data/processed/<dataset_name>/<model_name>/<template>/` 目录下：
- `concept_train.json` 和 `concept_train.csv`
- `concept_test.json` 和 `concept_test.csv`

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

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式: all/train/test | all |
| `--timestamp` | 测试模式的实验时间戳 | - |
| `--dataset_name` | 数据集名称 (TOXICN/COLD) | TOXICN |
| `--model_name` | LLM模型名称 | Qwen2.5-1.5B-Instruct |
| `--template` | 提示词模板类型 (binary/likert) | binary |
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

## 实验输出

```
experiments/<timestamp>/
├── config.json              # 实验配置快照
├── best_model.pth         # 最佳模型权重
├── metrics.png            # 训练曲线图
└── test_results/         # 测试结果
    ├── metrics.json     # 评估指标
    ├── classification_report.txt  # 详细分类报告
    └── predictions.json   # 逐条预测结果
```

## 数据集

| 数据集 | 说明 |
|--------|------|
| TOXICN | 中文毒性言论数据集 |
| COLD | 中文冒犯性语言数据集 |

原始数据格式：
```json
[{"topic": "race", "content": "文本内容", "toxic": 1, "toxic_one_hot": [0, 1]}]
```

概念向量数据格式：
```json
[{"content": "文本内容", "toxic": 1, "concept": [0.85, 0.12, 0.93, ..., 0.45]}]
```

## 评估指标

所有指标采用 **Macro Average**（宏平均）：
- **Precision**：预测为有毒的样本中真正有毒的比例
- **Recall**：真正有毒的样本中被正确预测的比例
- **F1 Score**：精确率和召回率的调和平均数

## 环境要求

```
Python >= 3.8
PyTorch >= 2.0
transformers >= 4.50
scikit-learn >= 1.0
matplotlib >= 3.5
pandas >= 1.5
vllm >= 0.4.0  # 用于概念向量生成
```

安装依赖：
```bash
pip install -r requirements.txt
```

## 可复现性

支持 `--use_deterministic` 参数启用确定性训练模式，配合 `--seed` 确保实验可复现。

## License

本项目仅供学术研究使用。
