# toxicnew

基于**类型感知概念瓶颈模型**的中文有害言论检测系统。

本仓库为项目当前主分支：聚焦 **v4 类型感知多概念向量**（`generate_typed_concept_vllm.py`），将上游 LLM 生成的概念向量经可解释概念瓶颈层送入分类器，实现**性能与内在可解释性兼备**的有害言论检测。

> 历史版本：v1（5级Likert）脚本仍保留于 `scripts/` 供参考，但已不参与下游流程；v2 三级行为锚点版本已独立迁移至 `d:\fscbm` 并形成论文 ACV-FSCBM（ToxiCN 加权 F1=0.817）。

## 整体流程

```
原始文本 → 类型感知概念向量生成(vLLM) → 特征提取(concept_features) → MLP分类器 → 有毒/无毒
```

## 项目结构

```
toxicnew/
├── configs/
│   └── MLP_config.py              # 训练超参数与路径配置
├── models/
│   └── mlp.py                     # MLP / TypeAugmentedGatedMLP 模型
├── scripts/
│   ├── generate_typed_concept_vllm.py          # ★ 类型感知概念向量生成（当前主脚本）
│   ├── inspect_typed_verbalizer_coverage_vllm.py # 类型化提示词 verbalizer 覆盖率验证
│   ├── generate_adjective_c_r_vllm.py           # 历史 v1 版本（仅供参考，已弃用）
│   └── inspect_verbalizer_coverage_vllm.py      # 历史 v1 verbalizer 覆盖率（仅供参考）
├── utils/
│   ├── mlp_pipeline.py            # MLP 训练与测试统一流水线
│   ├── concept_features.py        # typed 概念向量特征提取（single/conditional/all_probs）
│   └── seed.py                    # 随机种子与确定性配置
├── data/
│   ├── raw/                       # 原始数据集与概念词典
│   └── processed/                 # 生成的概念向量
├── experiments/                   # 实验结果
├── docs/prompts/                  # 项目背景提示词（供 AI 上下文）
└── requirements.txt
```

## 类型感知概念向量生成

`generate_typed_concept_vllm.py` 基于 v4 混合概念词典（134 概念，7 种语义类型），为不同概念类型使用**专属提示词模板与 verbalizer**，以从 LLM 中更充分提取已有信息。

| 类型 | 概念数 | Verbalizer | 标量分数 |
|------|--------|-----------|---------|
| behavior（行为型） | 27 | 二元 ["1","2"] | P(2) |
| strategy（策略型） | 25 | 二元 ["1","2"] | P(2) |
| evaluation（评价型） | 16 | 3级 ["1","2","3"] | P(3) |
| discrimination（歧视型） | 33 | 3级 ["1","2","3"] | P(3) |
| neutral（中性概念） | 6 | 3级 ["1","2","3"] | P(3) |
| intent（意图概念） | 15 | 二元 ["1","2"] | P(2) |
| effect（效果概念） | 12 | 二元 ["1","2"] | P(2) |

- 使用 vLLM 推理（max_tokens=1, temperature=0, logprobs=20）
- 从首 token logprobs 提取 verbalizer token 概率并做 softmax 归一化
- `concept_scores`：统一"有害/肯定"概率 [0,1]；`level_probs`：保留完整原始概率

## 特征提取

`concept_features.py` 依据 `concept_feat_mode` 从 `level_probs` 提取特征矩阵：

| 模式 | 二元类型特征 | 3级类型特征 | 说明 |
|------|------------|------------|------|
| single | P(2) | P(3) | 最简洁 |
| conditional | P(2) | P(3)+P(3)/(P(2)+P(3)) | 当前默认，默认 `concept_feat_mode=conditional` |
| all_probs | P(1),P(2) | P(1),P(2),P(3) | 保留全部信息，维度最高 |

## MLP 模型架构

- **MLP（GatedMLP，标准）**：门控(sigmoid) → Dropout → FC→96 → ReLU → Dropout → FC→2
- **TypeAugmentedGatedMLP**：在门控基础上追加类型级汇总特征（mean/max）

## 配置文件

训练超参数统一在 `configs/MLP_config.py` 中配置：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `concept_feat_mode` | 特征提取模式 | conditional |
| `model_type` | 模型类型（mlp/type_augmented_gated_mlp） | mlp |
| `batch_size` | 批次大小 | 16 |
| `epochs` | 训练轮数 | 200 |
| `max_lr` | 峰值学习率 | 1e-3 |
| `pct_start` / `div_factor` / `final_div_factor` / `anneal_strategy` | OneCycleLR 参数 | 0.2 / 25 / 10000 / cos |
| `dropout_rate` | Dropout 比率 | 0.5 |
| `hidden_features` | 隐藏层维度 | 96 |
| `patience` | 早停耐心值 | 20 |
| `seed` / `use_deterministic` | 随机种子 / 确定性模式 | 1 / False |

## 使用方法

**第一步：生成概念向量**

```bash
python scripts/generate_typed_concept_vllm.py --mode train --dataset_name TOXICN --model_name glm-4-9b-chat
python scripts/generate_typed_concept_vllm.py --mode test  --dataset_name TOXICN --model_name glm-4-9b-chat
```

生成文件位于 `data/processed/<dataset>/<model>/concept_{train,test}_<model>_typed_v4.json`。

**第二步：训练与测试**

```bash
# 训练 + 测试
python utils/mlp_pipeline.py --mode all --dataset_name TOXICN --model_name glm-4-9b-chat

# 仅训练
python utils/mlp_pipeline.py --mode train --dataset_name TOXICN --model_name glm-4-9b-chat

# 仅测试（需指定实验时间戳）
python utils/mlp_pipeline.py --mode test --dataset_name TOXICN --model_name glm-4-9b-chat --timestamp <时间戳>
```

## 命令行参数（mlp_pipeline.py）

| 参数 | 说明 | 必填 |
|------|------|------|
| `--mode` | all / train / test | 默认 all |
| `--dataset_name` | TOXICN / COLD | 是 |
| `--model_name` | 概念向量生成所用的 LLM | 是 |
| `--timestamp` | 测试模式的实验时间戳 | test 模式必填 |
| `--adjective_name` | 概念词典文件名 | 否，默认 toxic_adjectives_v4.csv |
| `--model_type` | mlp / type_augmented_gated_mlp | 否，默认 mlp |

参数优先级：训练模式 命令行 > MLP_config.py；测试模式强制使用实验目录中的 config.json。

## 实验输出

```
experiments/<timestamp>/
├── config.json              # 完整实验配置快照（概念向量 + 分类器 + 超参数 + 数据规模）
├── best_model.pth           # 最佳模型权重
├── metrics.png              # 训练曲线图
└── test_results/            # 测试结果
    ├── metrics.json         # 评估指标
    ├── classification_report.txt  # 详细分类报告
    └── predictions.json     # 逐条预测结果
```

`config.json` 记录"用了哪个概念向量、什么分类器"：`dataset_name/model_name/adjective_path/concept_feat_mode/num_concepts/concept_dim` 定位上游概念向量，`model_type/hidden_features/dropout_rate` 定位下游分类器。

## 数据集与概念词典

| 数据集 | 说明 |
|--------|------|
| TOXICN | 中文毒性言论数据集（训练 9600 / 测试 2411，87.4% 隐式毒性） |
| COLD | 中文冒犯性语言数据集（训练 32157 / 测试 5323） |

| 概念词典 | 概念数 | 说明 |
|----------|--------|------|
| toxic_adjectives_v4.csv | 134 | **当前主词典**，7 类型混合概念 |
| toxic_adjectives_v1.csv | 177 | 历史 v1，仅供参考 |

## 评估指标

下游分类器采用 **Macro Average**（宏平均）：Precision / Recall / F1。

## 环境要求

```
Python >= 3.8
PyTorch >= 2.0
transformers >= 4.50
scikit-learn >= 1.0
matplotlib >= 3.5
numpy
vllm
```

安装依赖：`pip install -r requirements.txt`

## License

本项目仅供学术研究使用。