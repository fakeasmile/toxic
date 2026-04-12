# 实验分支：experiment/emb-cat-proj

## 实验目标

将 BERT 嵌入层中 `toxic_emb` 和 `stance_emb` 的融合方式从**相加**改为**拼接+投影**，以保留各特征的独立表征能力。

## 基于主分支的改动

### 1. `models/bert.py`

#### 1.1 新增导入

```python
import torch.nn.functional as F

VALID_PROJ_TYPES = ("linear", "linear_norm", "linear_act_norm")
```

#### 1.2 `CustomEmbeddings` 类

**`__init__` 方法改动：**

新增参数 `proj_type="linear"`，控制投影层结构，支持三种选项：

| `proj_type` 值 | 投影层结构 | 说明 |
|---|---|---|
| `"linear"` | `Linear(3H→H)` | 选项a：纯线性投影 |
| `"linear_norm"` | `Linear(3H→H) → LayerNorm(H)` | 选项b：投影后归一化 |
| `"linear_act_norm"` | `Linear(3H→H) → GELU → LayerNorm(H)` | 选项c：投影后激活+归一化 |

- 新增 `assert` 校验 `proj_type` 合法性
- 保存 `self.proj_type` 供 `forward` 使用

新增投影层：

```python
self.projection = nn.Linear(config.hidden_size * 3, config.hidden_size)
```

拼接后维度为 `3 × 768 = 2304`，通过投影层映射回 `768`。

条件创建归一化层（仅选项 b/c）：

```python
if self.proj_type in ("linear_norm", "linear_act_norm"):
    self.proj_norm = nn.LayerNorm(config.hidden_size)
```

**`forward` 方法改动：**

| 改动点 | 主分支（相加） | 本分支（拼接+投影） |
|---|---|---|
| toxic_ids 缺失时 | 跳过，不处理 | 零向量填充 `torch.zeros_like(embeddings)` |
| stance_ids 缺失时 | 跳过，不处理 | 零向量填充 `torch.zeros_like(embeddings)` |
| 融合方式 | `embeddings += toxic_emb` + `embeddings += stance_emb` | `torch.cat([embeddings, toxic_emb, stance_emb], dim=-1)` + `self.projection(concat_emb)` |
| 投影后处理 | 无 | 按 `proj_type` 条件执行 GELU / LayerNorm |

完整 forward 流程：
```python
# 1. BERT原始嵌入
embeddings = super().forward(input_ids, token_type_ids, position_ids)

# 2. 毒性嵌入（缺失用零向量）
toxic_emb = self.toxic_embeddings(toxic_ids) if toxic_ids is not None else torch.zeros_like(embeddings)

# 3. 攻击立场嵌入（缺失用零向量）
stance_emb = self.stance_embeddings(stance_ids) if stance_ids is not None else torch.zeros_like(embeddings)

# 4. 拼接 + 线性投影
concat_emb = torch.cat([embeddings, toxic_emb, stance_emb], dim=-1)  # [B, seq_len, 2304]
embeddings = self.projection(concat_emb)                               # [B, seq_len, 768]

# 5. 选项c: GELU激活
if self.proj_type == "linear_act_norm":
    embeddings = F.gelu(embeddings)

# 6. 选项b/c: LayerNorm归一化
if self.proj_type in ("linear_norm", "linear_act_norm"):
    embeddings = self.proj_norm(embeddings)
```

#### 1.3 `ModifiedBert` 类

**`__init__` 方法改动：**

- 新增参数 `proj_type="linear"`
- 透传给 `CustomEmbeddings`：

```python
# 主分支
CustomEmbeddings(config, embeddings, num_toxic_types)

# 本分支
CustomEmbeddings(config, embeddings, num_toxic_types, proj_type)
```

`forward` 方法无改动。

---

### 2. `configs/base_config.py`

新增配置项：

```python
# 嵌入层投影类型: "linear"/"linear_norm"/"linear_act_norm"
self.proj_type = "linear_norm"
```

当前默认值为 `"linear_norm"`（选项b），可通过修改此值切换投影方式。

---

### 3. `utils/train.py`

#### 3.1 `init()` 函数

config.json 保存内容新增 `proj_type` 字段：

```python
config_dict = {
    ...
    "num_toxic_types": base_config.num_toxic_types,
    "proj_type": base_config.proj_type,  # 新增
}
```

#### 3.2 `final_train()` 函数

`ModifiedBert` 初始化新增 `proj_type` 参数：

```python
# 主分支
bert_model = ModifiedBert(bert_path, freeze_bert_layers, num_toxic_types).to(device)

# 本分支
bert_model = ModifiedBert(bert_path, freeze_bert_layers, num_toxic_types, proj_type).to(device)
```

---

### 4. `utils/test.py`

#### 4.1 配置读取

从 config.json 读取 `proj_type`，兼容旧实验目录（默认 `"linear"`）：

```python
proj_type = config.get("proj_type", "linear")  # 兼容旧实验目录
```

#### 4.2 模型初始化

```python
# 主分支
bert_model = ModifiedBert(bert_path, freeze_bert_layers, num_toxic_types).to(device)

# 本分支
bert_model = ModifiedBert(bert_path, freeze_bert_layers, num_toxic_types, proj_type).to(device)
```

---

## 未改动的文件

| 文件 | 说明 |
|---|---|
| `models/fc.py` | FC 分类器输入仍为 768 维，无需改动 |
| `models/mlp.py` | MLP 模型不涉及 BERT 嵌入层 |
| `Pure_Bert` 类 | 纯 BERT 模型，不使用自定义嵌入层 |

## 维度变化总结

```
主分支：  embeddings(768) += toxic_emb(768) += stance_emb(768) → 768 → Encoder
本分支：  cat[embeddings(768), toxic_emb(768), stance_emb(768)] → 2304 → Projection → 768 → Encoder
```

## 切换投影方式

修改 `configs/base_config.py` 中的 `self.proj_type`：

| 值 | 效果 |
|---|---|
| `"linear"` | 选项a：纯线性投影 |
| `"linear_norm"` | 选项b：线性投影 + LayerNorm（当前默认） |
| `"linear_act_norm"` | 选项c：线性投影 + GELU + LayerNorm |
