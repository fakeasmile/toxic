# H-CBM 层次化概念瓶颈模型使用指南

## 一、方案概述

H-CBM（Hierarchical Concept Bottleneck Model）是针对中文有害言论检测提出的层次化概念瓶颈模型。核心创新是将单一的概念层扩展为**三层递进结构**：

- **修辞层（Rhetoric）**：识别反讽、隐喻、夸张、狗哨政治等修辞手法
- **情感层（Emotion）**：识别愤怒、蔑视、敌意、嘲讽等情感态度
- **语义层（Semantic）**：识别辱骂、歧视、威胁、煽动等语义内容

三层之间采用**门控自适应层次融合**：各层概念保持原始维度，通过一个层次门控网络学习每个样本的三层重要性权重（如修辞层0.2、情感层0.3、语义层0.5），实现样本自适应的层次信息加权。

---

## 二、文件结构

```
toxicnew/
├── data/raw/adjective/
│   ├── hcbm_rhetoric_concepts.csv    # 修辞层形容词词表（20个）
│   ├── hcbm_emotion_concepts.csv     # 情感层形容词词表（22个）
│   └── hcbm_semantic_concepts.csv    # 语义层形容词词表（85个）
├── models/
│   ├── mlp.py                        # 原始MLP基线模型
│   └── hierarchical_mlp.py           # H-CBM层次化模型 + Flat基线
├── scripts/
│   ├── generate_adjective_c_r_vllm.py   # 原始单层概念生成
│   └── generate_hcbm_concepts.py        # H-CBM三层概念生成
├── utils/
│   ├── mlp_pipeline.py               # 原始MLP训练流水线
│   ├── hcbm_pipeline.py              # H-CBM训练流水线
│   ├── hcbm_intervention.py          # 概念干预与反事实解释
│   └── hcbm_visualize.py             # 可视化与可解释性分析
└── configs/
    └── MLP_config.py                 # 配置文件（已扩展H-CBM参数）
```

---

## 三、使用流程

### 步骤1：生成三层概念向量

使用 `scripts/generate_hcbm_concepts.py` 分别为训练集和测试集生成三层概念向量。

```bash
# 训练集
python scripts/generate_hcbm_concepts.py \
    --mode train \
    --dataset_name TOXICN \
    --model_name Qwen2.5-7B-Instruct-AWQ \
    --template likert \
    --quantization awq \
    --gpu_memory_utilization 0.85

# 测试集
python scripts/generate_hcbm_concepts.py \
    --mode test \
    --dataset_name TOXICN \
    --model_name Qwen2.5-7B-Instruct-AWQ \
    --template likert \
    --quantization awq \
    --gpu_memory_utilization 0.85
```

**输出位置**：`data/processed/TOXICN/Qwen2.5-7B-Instruct-AWQ/likert/`
- `concept_rhetoric_train.json` / `concept_rhetoric_test.json`
- `concept_emotion_train.json` / `concept_emotion_test.json`
- `concept_semantic_train.json` / `concept_semantic_test.json`

---

### 步骤2：训练H-CBM模型

使用 `utils/hcbm_pipeline.py` 训练模型。

```bash
# 训练H-CBM v2层次化模型（推荐）
python utils/hcbm_pipeline.py \
    --mode all \
    --dataset_name TOXICN \
    --model_name Qwen2.5-7B-Instruct-AWQ \
    --template likert \
    --epochs 300 \
    --max_lr 5e-4 \
    --hidden_features 128 \
    --patience 30 \
    --use_deterministic \
    --seed 42

# 训练扁平拼接基线（消融实验对比）
python utils/hcbm_pipeline.py \
    --mode all \
    --dataset_name TOXICN \
    --model_name Qwen2.5-7B-Instruct-AWQ \
    --template likert \
    --use_flat_baseline \
    --epochs 300 \
    --max_lr 5e-4 \
    --hidden_features 128 \
    --patience 30
```

**关键参数**：
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--hidden_features` | 隐藏层维度 | 128 |
| `--dropout_rate` | Dropout比率 | 0.3 |
| `--use_flat_baseline` | 使用扁平拼接基线 | False |

**输出位置**：`experiments/hcbm_<timestamp>/`
- `config.json`：实验配置
- `best_model.pth`：最佳模型权重
- `metrics.png`：训练曲线
- `test_results/`：测试结果

---

### 步骤3：概念干预（Test-time Intervention）

使用 `utils/hcbm_intervention.py` 进行概念级干预。

```bash
# 单概念干预：将样本5的语义层"辱骂性"概念设为0
python utils/hcbm_intervention.py \
    --timestamp hcbm_20260510-120000 \
    --mode intervene \
    --sample_idx 5 \
    --layer semantic \
    --concept_idx 0 \
    --new_value 0.0

# 反事实解释：样本0为什么被判定为有毒？
python utils/hcbm_intervention.py \
    --timestamp hcbm_20260510-120000 \
    --mode counterfactual \
    --sample_idx 0 \
    --top_k 5

# 批量反事实解释（所有有毒样本）
python utils/hcbm_intervention.py \
    --timestamp hcbm_20260510-120000 \
    --mode batch_counterfactual \
    --top_k 5 \
    --output_path experiments/hcbm_20260510-120000/test_results/batch_cf.json
```

---

### 步骤4：可视化分析

使用 `utils/hcbm_visualize.py` 生成可解释性可视化。

```bash
# 单样本概念激活热力图
python utils/hcbm_visualize.py \
    --timestamp hcbm_20260510-120000 \
    --mode heatmap \
    --sample_idx 0

# 全局概念重要性排序
python utils/hcbm_visualize.py \
    --timestamp hcbm_20260510-120000 \
    --mode importance \
    --top_n 15

# 层次间信息流相关性
python utils/hcbm_visualize.py \
    --timestamp hcbm_20260510-120000 \
    --mode flow

# 混淆矩阵
python utils/hcbm_visualize.py \
    --timestamp hcbm_20260510-120000 \
    --mode confusion

# 各层独立分类能力对比（消融实验）
python utils/hcbm_visualize.py \
    --timestamp hcbm_20260510-120000 \
    --mode layer_comparison
```

**输出位置**：`experiments/hcbm_<timestamp>/visualizations/`

---

## 四、与原始代码的对比

| 维度 | 原始SCBM | H-CBM v2 |
|------|---------|----------|
| 概念层 | 单层（177个形容词） | 三层（修辞/情感/语义） |
| 模型架构 | 门控MLP | 门控MLP + 层次门控自适应融合 |
| 可解释性 | 全局+局部概念权重 | + 层次重要性权重 + 概念干预 + 反事实解释 |
| 概念关系 | 独立 | 样本自适应的层次加权融合 |
| 训练脚本 | `utils/mlp_pipeline.py` | `utils/hcbm_pipeline.py` |
| 概念生成 | `scripts/generate_adjective_c_r_vllm.py` | `scripts/generate_hcbm_concepts.py` |

---

## 五、实验设计建议

### 5.1 基线对比实验

1. **原始SCBM**：使用 `utils/mlp_pipeline.py` 在TOXICN上运行（F1=0.7835）
2. **Flat-CBM**：使用 `utils/hcbm_pipeline.py --use_flat_baseline`（三层概念直接拼接）
3. **H-CBM**：使用 `utils/hcbm_pipeline.py`（完整层次化模型）

### 5.2 消融实验

- 仅修辞层：`hcbm_visualize.py --mode layer_comparison`
- 仅情感层：同上
- 仅语义层：同上
- 去掉层次门控：对比 Flat-CBM（无层次门控）与 H-CBM v2（有层次门控）

### 5.3 跨数据集验证

在COLD和SWSR数据集上重复步骤1-2，验证泛化能力。

---

## 六、常见问题

**Q1: 三层概念向量生成太慢怎么办？**
- 三层概念向量可以复用同一个vLLM模型实例，脚本已优化为只加载一次模型
- 可以考虑使用更小的LLM（如Qwen2.5-1.5B-Instruct）先生成概念向量，再用H-CBM训练

**Q2: 显存不足怎么办？**
- 降低 `--gpu_memory_utilization`（如0.7）
- 减小 `--hidden_features`（如64）
- 使用 `--batch_size 8`

**Q3: 如何调试概念向量质量？**
- 检查 `data/processed/.../concept_rhetoric_train.csv` 中的概念值分布
- 正常情况应有部分概念值>0.5（表示相关），部分接近0（表示无关）
- 如果所有值都接近0，可能是verbalizer或prompt问题

---

## 七、学术贡献点

1. **层次化概念空间**：首次将概念瓶颈模型扩展为三层递进结构，对应中文有害言论的修辞-情感-语义递进识别
2. **门控自适应层次融合**：通过层次门控网络学习样本自适应的三层重要性权重，避免跨层压缩导致的信息瓶颈
3. **双层可解释性**：同时提供概念级门控权重（哪些概念重要）和层次级门控权重（哪些层次重要）
4. **概念干预能力**：支持test-time intervention和counterfactual explanation
5. **中文语境适配**：三层概念词表专门针对中文网络语言特点设计（反讽、阴阳怪气、谐音等）
