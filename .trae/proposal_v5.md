# 有害言论检测 × 可解释AI — 深度调研与创新方案（v5）

> 调研日期：2026-06-01
> 基于：main 基线 + 11 个 feature 分支实验复盘 + 2024-2026 顶刊最新文献（ICLR 2025/2026, NeurIPS 2024/2025, CVPR 2024/2025, ECCV 2024, EMNLP 2025, ICML 2024）
> 数据集：TOXICN / COLD / SWSR
> 核心突破：**事后概念提取 + 句子嵌入概念评分 + 迭代概念完备 + 直接线性分类**

---

## 一、调研总结：2024-2026 顶刊前沿趋势

### 1.1 NLP 有害言论 + 可解释 AI 领域关键进展

| 论文 | 会议 | 核心贡献 | 与本项目相关性 |
|------|------|---------|--------------|
| MCBM | ICLR 2026 | 证明 CBM 概念瓶颈从未真正存在，提出 IB 约束 | 直接解释所有 CBM 方案失败的根因 |
| CREAM | ICLR 2026 | 概念推理图 + 正则化侧通道 | 解决扁平概念无法建模层次结构的问题 |
| CB-LLM | ICLR 2025 | LLM 概念瓶颈 + **ACC 自动概念修正** + 稀疏线性层 | **ACC 步骤和句子嵌入替代 Likert 评分** |
| BC-LLM | NeurIPS 2025 | 贝叶斯迭代概念搜索 + LLM 先验 | 迭代概念发现优于一次性定义 |
| CT-CBM | EMNLP 2025 Findings | 概念完整性 + 迭代发现直到完备 | 56 个概念可能既不完整也不精简 |
| SCBM | IPM 2025 | 形容词概念瓶颈 + Transformer 融合 | 在 CONAN（反言论数据集）上失败，与 TOXICN 失败模式相似 |
| MMBERT | AAAI 2026 | 多模态 MoE，TOXICN F1=0.922 | 性能天花板参考，但不可解释 |
| STATE ToxiCN | 2025 | span 级标注 + 编码术语词表 | 编码术语注入是突破 0.83 的关键外部知识 |

### 1.2 CV 领域 CBM 方法（可迁移到 NLP）

| 论文 | 会议 | 核心贡献 | 迁移价值 |
|------|------|---------|---------|
| **PCBM / PCBM-h** | **ICLR 2024** | **事后转化 CBM：先训练黑盒，再提取概念方向** | ★★★★★ 根本性范式转变 |
| **Label-free CBM** | **ICLR 2024** | **无需概念标注，CLIP 零样本概念评分** | ★★★★★ 句子嵌入替代 LLM Likert |
| **Concept2CLIP** | **CVPR 2024** | **对抗训练防止概念泄漏** | ★★★★★ 解决概念架空 |
| **DCR** | **ECCV 2024** | **直接概念回归：仅线性映射，从根本上防绕过** | ★★★★★ 最强防绕过方案 |
| **Intervene-CBM** | **NeurIPS 2024** | **干预感知训练：确保概念被模型真正使用** | ★★★★☆ |
| **Concept SAE** | **NeurIPS 2024** | **SAE + CBM 融合，SAE 特征作为概念层** | ★★★★★ |
| IB-CBM | ICML 2024 | 信息瓶颈约束防止概念冗余和泄漏 | ★★★★☆ |
| Causal CBM | ICML 2024 | 因果推理确保概念-标签因果关系 | ★★★★☆ |
| Autogenerate CBM | AAAI 2024 | 自动概念生成与质量评估 | ★★★★☆ |
| Minimal Sufficient Concepts | CVPR 2024 | 学习最小充分概念集 | ★★★★☆ |
| SALF-CBM | CVPR 2025 | 空间感知无标签 CBM，超越 backbone | ★★★☆☆ |
| Concept Strictness | ICCV 2025 | 概念严格度控制 | ★★★☆☆ |

### 1.3 中文隐式毒性检测最新进展

| 方法/数据集 | 年份 | 核心发现 |
|------------|------|---------|
| PCR-ToxiCN | 2025 | LLM 在拼音伪装文本上仅 F1=0.672，谐音检测是关键短板 |
| STATE ToxiCN | 2025 | 编码仇恨术语词表注入显著提升检测，LLM 在编码术语识别上表现差 |
| "Cracking the Code" | 2025 | 6 种编码策略分类（谐音/字形/语义伪装/反讽/标签化/暗语）作为辅助任务有效 |
| 融合仇恨对象+变体词还原 | 2026 | ChatGLM2-6B 变体词还原模块降低漏报率 3.45% |

---

## 二、SCBM 在中文 TOXICN 上失效的根本原因

### 2.1 核心矛盾：言外之意 vs 言内之意

SCBM 的 Likert 评分机制假设：**毒性可以通过文本与一组预定义形容词的相关性来表征**。这个假设在英语中基本成立，但在中文中系统性失效。

**英语**：有毒文本通常包含显式侮辱词（"You're an idiot"），与 "offensive"、"aggressive" 等形容词有直接语义映射。LLM 能准确评估相关性。

**中文**：大量有毒文本不包含任何显式侮辱词，毒性完全依赖语境、文化知识和语用推理。例如：

| 文本 | 字面含义 | 实际毒性 | LLM Likert 评分倾向 |
|------|---------|---------|-------------------|
| "您可真厉害呢" | 赞美 | 反讽贬低 | "攻击性的"→1-2分（低） |
| "就这？" | 疑问 | 轻蔑否定 | "蔑视的"→1-2分（低） |
| "呼叫腕豪！" | 游戏术语 | 性别侮辱（腕豪=贴吧女性蔑称） | "厌女的"→1分（极低） |
| "抓虫大战，人人有责" | 除虫活动 | 种族侮辱（虫=对特定种族的蔑称） | "种族偏见的"→1分（极低） |

**根本矛盾**：中文隐式毒性的本质是"言外之意"，而 SCBM 的 Likert 评分机制只能捕捉"言内之意"。

### 2.2 五重结构性错配

| 维度 | 英文数据集（ELF22/HS-CS） | 中文 TOXICN | 错配后果 |
|------|--------------------------|------------|---------|
| 毒性表达 | 显式为主（70%+含侮辱词） | 隐式为主（52.7%无显性表达） | 形容词-毒性映射弱 |
| 概念空间 | 情感态度维度足够 | 需"目标×策略×意图"多维结构 | 扁平形容词概念不够 |
| LLM 能力 | 英语语用推理成熟 | 中文语用推理不足 | Likert 评分系统性偏低 |
| 外部知识 | 预训练语料覆盖好 | 缺少中文网络黑话/谐音知识 | 模型无法识别编码术语 |
| 监督信号 | Likert 评分可靠 | Likert 评分不可靠 | 概念监督质量差 |

### 2.3 SCBM 在 CONAN 上的失败印证

SCBM 原论文中，在 CONAN（反言论数据集，需语境推理）上 BERT-base F1=0.721，而 SCBM 仅 0.630——**大幅落后 BERT 11.2%**。这与 TOXICN 上的失败模式高度相似：当数据集需要语用推理而非字面匹配时，LLM 概念评分系统性失效。

### 2.4 之前 11 个方案的统一失败模式

所有方案可归为三类失败：

**类型 A：概念层信息压缩过度**
- lrd-cbm：1536→56 Sigmoid 瓶颈，损失 >99% 信息 → F1=0.8168
- cgrm：概念-Token 注意力无显式监督 → F1≈0.79

**类型 B：残差路径使概念层被架空**
- cotd-cbm：768 维残差 concat，分类器直接读原始表示 → F1=0.8205
- ki-scb：概念 9→128 vs 残差 1536→128，概念信号被淹没 → F1=0.8269
- rcwn：残差 768 维 vs 概念 56 维 → F1=0.7851

**类型 C：概念监督信号质量不足**
- main：纯 LLM Likert，对隐式毒性评分偏低 → F1=0.7823
- hecd-net：概念嵌入从 [CLS] 学习，无新信息 → F1=0.8165
- ttt-cbm：MLM 目标与毒性分类不一致 → F1=0.8145

**统一根因（MCBM, ICLR 2026）**：概念瓶颈从未真正存在——概念层可以编码 nuisance 信息，残差路径提供"后门"，概念层被架空是必然结果。

---

## 三、新方案：KIPC-CBM（Knowledge-Informed Post-hoc Concept Bottleneck Model）

### 3.1 核心设计哲学

> **"先强后释"——先训练最强黑盒，再事后提取可解释概念**

之前所有方案的根本问题：**在训练过程中同时优化性能和可解释性，模型总是选择性能而牺牲可解释性**。概念层被架空不是 bug，而是训练动态的必然结果——当模型可以通过残差路径获得更丰富的信息时，它没有理由依赖概念层。

**范式转变**：

| 维度 | 之前所有方案（联合训练范式） | KIPC-CBM（事后提取范式） |
|------|--------------------------|------------------------|
| 训练策略 | 概念层 + 分类器联合训练 | 先训练黑盒，再提取概念 |
| 概念来源 | 人工预定义 + LLM 评分 | 从训练好的黑盒中提取 |
| 概念-表示对齐 | 不保证（概念可能是模型无法检测的） | 天然对齐（概念来自模型自身表示） |
| 性能-可解释性权衡 | 训练中博弈，可解释性总是输 | 分离：先保证性能，再保证可解释性 |
| 概念充分性 | 无法验证 | 可验证：概念模型 vs 黑盒模型的性能差距 |
| 防绕过机制 | IB 约束（但训练中可能被绕过） | 直接线性分类 + 无残差路径 |

### 3.2 整体架构

```
==================== Phase 1: 黑盒训练 ====================

输入文本
    ↓
[知识预处理层]
    ├── 编码术语匹配：STATE ToxiCN 词表 → coded_mask [B, seq_len]
    ├── 谐音还原：homo_graph 映射 → restored_text
    └── 平台/主题嵌入：metadata → context_emb [B, 64]
    ↓
[双路 RoBERTa]（共享权重）
    ├── RoBERTa(original_text) → h_orig [B, 768]
    └── RoBERTa(restored_text) → h_rest [B, 768]
    ↓
[知识融合层]
    ├── h_fused = concat([h_orig, h_rest]) [B, 1536]
    ├── h_fused += Linear(coded_features)          ← 编码术语注入
    └── h_fused *= (1 + Linear(platform_emb))      ← 平台条件化
    ↓
[多任务分类头]
    ├── toxic_head: Linear(1536, 2) → 有毒/无毒
    ├── topic_head: Linear(1536, 5) → race/gender/region/lgbt/none
    └── expression_head: Linear(1536, 3) → direct/indirect/other
    ↓
目标：Macro F1 > 0.84（黑盒基线）


==================== Phase 2: 事后概念提取 ====================

训练好的黑盒模型（冻结）
    ↓
提取 [CLS] 表示：h_fused [B, 1536]
    ↓
[三路概念提取]
    │
    ├── [S 路：结构化概念探测]（有监督）
    │   ├── 对每个结构化概念 j（target×5 + strategy×3 + intent×5 + tone×4 = 17个）
    │   ├── 训练线性探针：ĉ_j = sigmoid(w_j^T @ h_fused + b_j)
    │   └── 监督信号：TOXICN 标注（topic, expression）+ LLM 分类伪标签（intent, tone）
    │
    ├── [A 路：句子嵌入概念评分]（零样本，CB-LLM 启发）
    │   ├── 对每个形容词概念 j（56个 discovered_concepts）
    │   ├── concept_emb_j = SentenceEncoder(concept_j_definition)
    │   ├── text_emb = SentenceEncoder(text)
    │   └── ĉ_j = cosine_similarity(text_emb, concept_emb_j)
    │
    └── [D 路：SAE 发现概念]（无监督，DN-CBM 启发）
        ├── 在 h_fused 上训练 G-SAE（离线）
        ├── h_sparse = TopK(W_enc @ h_fused + b_enc) [B, K]
        ├── 筛选毒性相关特征（t-test + 重要性评分）
        └── ĉ_D = h_sparse[:, selected_indices] [B, D]
    ↓
[概念向量] c = concat([ĉ_S, ĉ_A, ĉ_D]) [B, 17+56+D]


==================== Phase 3: 概念分类训练 ====================

概念向量 c [B, 17+56+D]
    ↓
[自动概念修正（ACC）]（CB-LLM 启发）
    ├── 计算每个概念与毒性标签的互信息
    ├── 移除零/负相关概念（噪声概念）
    └── 保留正相关的概念子集 c' [B, M]（M ≤ 17+56+D）
    ↓
[直接线性分类]（DCR 启发）
    ├── logits = W_sparse @ c' + b [B, 2]
    ├── L1 稀疏约束：||W_sparse||_1
    └── 无残差路径，无 MLP，无隐藏层
    ↓
[IB 约束]（MCBM 启发，可选）
    ├── 对 S 路概念探针施加 KL 约束
    └── 确保概念编码器只保留概念相关信息
    ↓
[迭代概念完备]（CT-CBM / BC-LLM 启发）
    ├── 如果 F1_concept < F1_blackbox - δ：
    │   ├── 分析错误样本，识别缺失概念维度
    │   ├── 增加 SAE 发现特征或生成新概念方向
    │   └── 重新训练概念分类器
    └── 直到 F1_concept ≈ F1_blackbox
    ↓
[输出] → 分类结果 + 概念解释 + 概念干预
```

### 3.3 核心模块详解

#### 模块 A：知识增强黑盒分类器（Phase 1）

**目标**：训练一个 F1 > 0.84 的黑盒分类器，作为概念提取的基础。

**架构**：双路 RoBERTa + 知识融合 + 多任务学习

```python
class KnowledgeEnhancedClassifier(nn.Module):
    def __init__(self, roberta_path, num_coded_terms=500, num_platforms=2):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(roberta_path)
        self.coded_term_emb = nn.Embedding(num_coded_terms + 1, 64)
        self.platform_emb = nn.Embedding(num_platforms + 1, 32)
        self.coded_proj = nn.Linear(64, 1536)
        self.platform_proj = nn.Linear(32, 1536)
        self.toxic_head = nn.Linear(1536, 2)
        self.topic_head = nn.Linear(1536, 5)
        self.expression_head = nn.Linear(1536, 3)

    def forward(self, input_ids_orig, attention_mask_orig,
                input_ids_rest, attention_mask_rest,
                coded_term_ids, platform_ids):
        h_orig = self.roberta(input_ids_orig, attention_mask_orig).last_hidden_state[:, 0]
        h_rest = self.roberta(input_ids_rest, attention_mask_rest).last_hidden_state[:, 0]
        h_fused = torch.cat([h_orig, h_rest], dim=-1)

        coded_emb = self.coded_proj(self.coded_term_emb(coded_term_ids))
        platform_gate = 1 + self.platform_proj(self.platform_emb(platform_ids))
        h_fused = h_fused + coded_emb
        h_fused = h_fused * platform_gate

        toxic_logits = self.toxic_head(h_fused)
        topic_logits = self.topic_head(h_fused)
        expression_logits = self.expression_head(h_fused)
        return toxic_logits, topic_logits, expression_logits, h_fused
```

**训练策略**：
- 损失：`L = CE(toxic) + 0.3 * CE(topic) + 0.3 * CE(expression)`
- 差分学习率：RoBERTa 2e-5, 分类头 1e-3
- 对抗训练：FGM（可选，提升鲁棒性）
- 早停基于验证集 Macro F1

**关键验证**：如果黑盒 F1 < 0.84，说明知识注入方案需要调整，不应进入 Phase 2。

#### 模块 B：三路概念提取（Phase 2）

**这是方案的核心创新——从训练好的黑盒中提取概念，而非在训练中强加概念。**

##### B1：结构化概念探测（S 路，有监督）

利用 TOXICN 的丰富标注（topic, expression）和 LLM 分类伪标签（intent, tone），训练线性探针从 h_fused 中提取结构化概念。

```python
class StructuredConceptProbe(nn.Module):
    def __init__(self, input_dim=1536, num_concepts=17):
        super().__init__()
        self.probes = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(num_concepts)
        ])

    def forward(self, h_fused):
        concept_scores = []
        for probe in self.probes:
            concept_scores.append(torch.sigmoid(probe(h_fused)))
        return torch.cat(concept_scores, dim=-1)
```

**17 个结构化概念**：

| 组别 | 概念 | 数量 | 监督来源 |
|------|------|------|---------|
| 目标群体 | race, gender, region, lgbt, none | 5 | TOXICN topic 标注 |
| 表达策略 | direct, indirect, other | 3 | TOXICN expression 标注 |
| 攻击意图 | degrade, discriminate, incite, objectify, none | 5 | LLM 分类伪标签 |
| 情感基调 | hostile, contemptuous, callous, neutral | 4 | LLM 分类伪标签 |

**为什么线性探针而非 MLP**：
1. 线性探针保证概念是 h_fused 中的线性可分特征，可解释性更强
2. 避免非线性探针"记忆"训练标签（Alain & Bengio, 2016）
3. 与 DCR 的直接线性分类理念一致

**LLM 分类伪标签生成**（替代 Likert 评分）：

```python
prompt = """分析以下文本的攻击意图和情感基调。

文本：{text}

请从以下选项中选择（每类选一个）：
攻击意图：[贬低/歧视/煽动/物化/无]
情感基调：[愤怒敌对/蔑视轻蔑/冷漠戏谑/中性]

以JSON格式输出：{{"intent": "...", "tone": "..."}}"""
```

**为什么分类伪标签比 Likert 评分更可靠**：
1. 分类任务比评分任务更稳定（选择"贬低" vs "歧视" 比评分"3分 vs 4分"更一致）
2. 选项是离散的，不存在 Likert 的边界模糊问题
3. LLM 在分类任务上的表现远优于回归/评分任务
4. 分类伪标签直接对应结构化概念，无需"形容词→结构化属性"的间接映射

##### B2：句子嵌入概念评分（A 路，零样本）

**这是从 CV 的 Label-free CBM 迁移到 NLP 的关键创新。**

在 CV 中，Label-free CBM 使用 CLIP 计算图像与概念文本描述的相似度作为概念得分。在 NLP 中，我们使用句子嵌入模型计算文本与概念定义的语义相似度。

```python
class SentenceEmbeddingConceptScorer:
    def __init__(self, model_name="BAAI/bge-base-zh-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name)

    def score(self, texts, concept_definitions):
        text_embs = self.encoder.encode(texts, normalize_embeddings=True)
        concept_embs = self.encoder.encode(concept_definitions, normalize_embeddings=True)
        similarity = text_embs @ concept_embs.T
        return (similarity + 1) / 2  # 映射到 [0, 1]
```

**关键优势**：
1. **零样本**：无需 LLM 推理，无需训练，直接计算
2. **确定性**：无采样噪声，结果可复现
3. **连续值**：无 Likert 1-5 的离散化信息损失
4. **更稳定**：句子嵌入捕获语义相似性，不受 LLM 提示工程影响
5. **高效**：批量编码，比 LLM 逐条推理快 10-100 倍

**对隐式毒性的处理**：
- 对原始文本和还原文本分别计算概念评分，取最大值
- 还原文本（谐音已解析）与毒性概念的相似度更高
- 这直接解决了 LLM 对隐式毒性评分偏低的问题

**概念定义**：使用 discovered_concepts.csv 中的 56 个概念及其定义（而非仅形容词名称），因为定义提供了更丰富的语义信息。

##### B3：SAE 发现概念（D 路，无监督）

**从 CV 的 Concept SAE / DN-CBM 迁移。**

在 h_fused 上训练 G-SAE（引导式稀疏自编码器），发现数据驱动的毒性相关特征。

```python
class GuidedSAE(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=4096, num_reserved=17, topk=64):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.num_reserved = num_reserved
        self.topk = topk

    def forward(self, h_fused, concept_labels=None):
        h_latent = self.encoder(h_fused)
        values, indices = torch.topk(h_latent, k=self.topk, dim=-1)
        h_sparse = torch.zeros_like(h_latent)
        h_sparse.scatter_(-1, indices, torch.sigmoid(values))

        h_reconstructed = self.decoder(h_sparse)
        loss_recon = F.mse_loss(h_reconstructed, h_fused)

        loss_guide = 0
        if concept_labels is not None:
            reserved = h_sparse[:, :self.num_reserved]
            loss_guide = F.binary_cross_entropy(
                reserved.clamp(1e-6, 1-1e-6), concept_labels
            )

        return h_sparse, loss_recon + 0.5 * loss_guide
```

**G-SAE 条件化**：预留前 17 个 latent index 给结构化概念（与 S 路对齐），确保 SAE 发现的特征包含结构化毒性属性。

**SAE 特征筛选**：
1. **类别区分度**：特征在有毒/无毒样本上的激活分布差异（t-test p < 0.05）
2. **概念重要性**：移除该特征后分类性能下降幅度
3. **可解释性**：用 Qwen2.5-7B 对高激活样本命名，人工抽检质量

**SAE 训练参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入维度 | 1536 | h_fused 维度 |
| 隐层维度 | 4096 | 过完备 2.67× 扩展 |
| TopK | 64 | 每样本最多激活 64 个特征 |
| Reserved indices | 17 | 结构化概念 |
| 训练时间 | ~15 分钟 | SAE 极轻量 |

#### 模块 C：自动概念修正（ACC）

**从 CB-LLM (ICLR 2025) 迁移的关键步骤——之前所有方案都缺少这一步。**

CB-LLM 论文发现：**ACC 是性能匹配黑盒的关键**。不做 ACC，CBM 性能远低于黑盒；做 ACC 后可匹配甚至超越黑盒。

```python
def automatic_concept_correction(concept_scores, labels, threshold=0.02):
    """
    自动概念修正：
    1. 计算每个概念与毒性标签的互信息
    2. 移除零/负相关概念（噪声概念）
    3. 保留正相关的概念子集
    """
    from sklearn.metrics import mutual_info_score
    selected_indices = []
    for j in range(concept_scores.shape[1]):
        mi = mutual_info_score(labels, (concept_scores[:, j] > 0.5).astype(int))
        if mi > threshold:
            selected_indices.append(j)
    return selected_indices
```

**ACC 的作用**：
1. **去除噪声概念**：某些形容词可能与毒性无关（如"简短的"、"直接的"），ACC 自动移除
2. **确保概念-标签对齐**：只保留与毒性正相关的概念，避免概念层编码无关信息
3. **降低概念维度**：从 17+56+D 维压缩到 M 维（M << 17+56+D），减轻过拟合

#### 模块 D：直接线性分类（DCR 启发）

**从 CV 的 DCR (ECCV 2024) 迁移——最强防绕过方案。**

```python
class DirectConceptRegression(nn.Module):
    def __init__(self, num_concepts, num_classes=2, l1_weight=0.01):
        super().__init__()
        self.linear = nn.Linear(num_concepts, num_classes)
        self.l1_weight = l1_weight

    def forward(self, concept_scores):
        logits = self.linear(concept_scores)
        return logits

    def l1_loss(self):
        return self.l1_weight * torch.norm(self.linear.weight, p=1)
```

**为什么直接线性分类是防绕过的最强方案**：
1. **无隐藏层**：不存在可以"记忆"信息的中间表示
2. **无残差路径**：分类器只能从概念得分中获取信息
3. **权重直接可解释**：`W[j, y]` 直接表示概念 j 对类别 y 的贡献
4. **L1 稀疏性**：每个预测只依赖少量概念，解释清晰

**与之前方案的对比**：

| 方案 | 分类层 | 可绕过性 | 可解释性 |
|------|--------|---------|---------|
| main | MLP(56→96→2) + 门控 | 高（门控可学习绕过） | 低 |
| cotd-cbm | Linear(824, 2) + 残差 | 极高（768 维残差后门） | 极低 |
| ki-scb | Linear(128, 2) + 残差 | 高（残差路径范数远大于概念路径） | 低 |
| **KIPC-CBM** | **Linear(M, 2) + L1** | **零（无残差，无隐藏层）** | **极高** |

#### 模块 E：迭代概念完备（CT-CBM / BC-LLM 启发）

**确保概念集充分覆盖黑盒模型使用的所有信息。**

```python
def iterative_concept_completion(blackbox_f1, concept_f1, delta=0.01):
    """
    迭代概念完备：
    1. 比较概念模型与黑盒模型的性能差距
    2. 如果差距 > delta，分析错误样本
    3. 增加 SAE 发现特征或生成新概念方向
    4. 重新训练概念分类器
    5. 重复直到性能差距 ≤ delta
    """
    iteration = 0
    while blackbox_f1 - concept_f1 > delta:
        iteration += 1
        error_analysis = analyze_errors(concept_model, blackbox_model, val_data)
        new_concepts = discover_new_concepts(error_analysis, sae_model)
        concept_scores = add_concepts(concept_scores, new_concepts)
        concept_model = train_concept_classifier(concept_scores, labels)
        concept_f1 = evaluate(concept_model, val_data)
    return concept_model
```

**新概念发现策略**：
1. **SAE 特征扩展**：增加 SAE 隐层维度或 TopK 值，发现更多特征
2. **错误聚类**：对概念模型错误分类的样本聚类，用 LLM 为每个聚类生成概念描述
3. **概念组合**：现有概念的交互特征（如 "地域 × 反讽"）

---

## 四、项目结构与运行指南

### 4.0 项目文件结构

```
toxicnew/
├── configs/
│   ├── MLP_config.py                          # 原有 MLP 配置
│   └── kipc_config.py                         # [新建] KIPC-CBM 全局配置
├── models/
│   ├── mlp.py                                 # 原有 MLP 模型
│   ├── blackbox_classifier.py                 # [新建] Phase 1: 知识增强黑盒分类器
│   ├── guided_sae.py                          # [新建] Phase 2b: G-SAE 模型
│   └── concept_classifier.py                  # [新建] Phase 3: 直接线性概念分类器
├── scripts/
│   ├── generate_adjective_c_r_vllm.py         # 原有 LLM Likert 评分脚本
│   ├── generate_pseudo_labels_vllm.py         # [新建] Phase 2a: LLM 分类伪标签生成
│   ├── extract_representations.py             # [新建] Phase 2b: 从黑盒提取 h_fused
│   ├── train_sae.py                           # [新建] Phase 2b: G-SAE 训练
│   ├── name_sae_features_vllm.py              # [新建] Phase 2b: LLM 对 SAE 特征命名
│   ├── compute_concept_scores.py              # [新建] Phase 2c: 三路概念评分计算
│   └── analyze_concepts.py                    # [新建] Phase 2d: ACC + 概念分析
├── utils/
│   ├── __init__.py
│   ├── mlp_pipeline.py                        # 原有 MLP 训练流水线
│   ├── seed.py                                # 原有随机种子工具
│   ├── blackbox_pipeline.py                   # [新建] Phase 1: 黑盒训练流水线
│   ├── concept_pipeline.py                    # [新建] Phase 3: 概念分类训练流水线
│   └── knowledge_utils.py                     # [新建] 编码术语匹配 + 谐音还原
├── data/
│   ├── raw/
│   │   ├── TOXICN/                            # TOXICN 数据集（含 topic/expression 标注）
│   │   ├── COLD/
│   │   ├── SWSR/
│   │   └── adjective/
│   │       ├── toxic_adjectives_new.csv       # 177 个形容词词表
│   │       ├── discovered_concepts.csv        # 56 个概念词表（含定义）
│   │       └── coded_terms.json               # [新建] 编码术语词表
│   └── processed/
│       └── TOXICN/
│           ├── Qwen2.5-7B-Instruct-AWQ/likert/ # 原有 LLM Likert 概念向量
│           └── kipc/                           # [新建] KIPC-CBM 中间数据
│               ├── representations/            # h_fused 表示
│               ├── pseudo_labels/              # LLM 伪标签
│               ├── sae/                        # SAE 模型 + 特征
│               └── concept_scores/             # 三路概念评分
├── models/                                     # 本地模型目录（.gitignore）
│   ├── chinese-roberta-wwm-ext/
│   ├── Qwen2.5-7B-Instruct-AWQ/
│   └── bge-base-zh-v1.5/                      # [需下载] 句子嵌入模型
└── experiments/                                # 实验结果
```

### 4.1 环境准备

```bash
# 1. 激活虚拟环境
conda activate TraeAI-3

# 2. 安装额外依赖（在原有 requirements.txt 基础上）
pip install sentence-transformers scikit-learn

# 3. 下载句子嵌入模型（如果本地没有）
# 方式一：手动下载到 models/ 目录
# 从 https://huggingface.co/BAAI/bge-base-zh-v1.5 下载到 models/bge-base-zh-v1.5/
# 方式二：首次运行时自动下载（需要网络）
```

### 4.2 Phase 1：黑盒训练

```bash
# 训练知识增强双路 RoBERTa 黑盒分类器
python utils/blackbox_pipeline.py \
    --mode all \
    --dataset_name TOXICN \
    --epochs 30 \
    --batch_size 16 \
    --patience 10 \
    --lr_backbone 2e-5 \
    --lr_head 1e-3 \
    --use_dual_encoder \
    --use_coded_terms \
    --use_homophone \
    --use_multitask \
    --lambda_topic 0.3 \
    --lambda_expression 0.3

# 仅测试已训练的黑盒模型
python utils/blackbox_pipeline.py \
    --mode test \
    --timestamp <实验时间戳>
```

**关键验证**：训练完成后检查验证集 Macro F1 是否 > 0.84。如果未达标，调整知识注入策略后重训。

**输出文件**：
- `experiments/<timestamp>/config.json` — 实验配置快照
- `experiments/<timestamp>/best_model.pth` — 最佳黑盒模型权重
- `experiments/<timestamp>/metrics.png` — 训练曲线图
- `experiments/<timestamp>/test_results/` — 测试结果

### 4.3 Phase 2：概念提取

#### Step 2a：LLM 伪标签生成

```bash
# 用 Qwen2.5-7B-Instruct-AWQ 生成 intent 和 tone 分类伪标签
python scripts/generate_pseudo_labels_vllm.py \
    --mode train \
    --dataset_name TOXICN \
    --model_name Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --gpu_memory_utilization 0.85

# 同样为测试集生成伪标签
python scripts/generate_pseudo_labels_vllm.py \
    --mode test \
    --dataset_name TOXICN \
    --model_name Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --gpu_memory_utilization 0.85
```

**输出文件**：
- `data/processed/TOXICN/kipc/pseudo_labels/pseudo_labels_train.json`
- `data/processed/TOXICN/kipc/pseudo_labels/pseudo_labels_test.json`

#### Step 2b：SAE 训练

```bash
# Step 2b-1: 从黑盒模型提取 h_fused 表示
python scripts/extract_representations.py \
    --dataset_name TOXICN \
    --blackbox_timestamp <Phase1的时间戳> \
    --mode all

# Step 2b-2: 训练 G-SAE
python scripts/train_sae.py \
    --dataset_name TOXICN \
    --sae_hidden_dim 4096 \
    --topk 64 \
    --num_reserved 17 \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-3 \
    --lambda_guide 0.5

# Step 2b-3: 用 LLM 对 SAE 特征命名
python scripts/name_sae_features_vllm.py \
    --dataset_name TOXICN \
    --model_name Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --sae_timestamp <SAE训练的时间戳> \
    --top_k_samples 10
```

**输出文件**：
- `data/processed/TOXICN/kipc/representations/h_fused_train.pt`
- `data/processed/TOXICN/kipc/representations/h_fused_test.pt`
- `data/processed/TOXICN/kipc/sae/sae_model.pth`
- `data/processed/TOXICN/kipc/sae/feature_names.json`

#### Step 2c：三路概念评分计算

```bash
# 一次性计算三路概念评分（S路 + A路 + D路）
python scripts/compute_concept_scores.py \
    --dataset_name TOXICN \
    --blackbox_timestamp <Phase1的时间戳> \
    --sae_timestamp <SAE训练的时间戳> \
    --mode all \
    --use_structured_probe \
    --use_sentence_embedding \
    --use_sae_discovery \
    --sentence_model_path models/bge-base-zh-v1.5 \
    --concept_definition_path data/raw/adjective/discovered_concepts.csv
```

**输出文件**：
- `data/processed/TOXICN/kipc/concept_scores/concept_scores_train.json`
- `data/processed/TOXICN/kipc/concept_scores/concept_scores_test.json`
- `data/processed/TOXICN/kipc/concept_scores/concept_metadata.json`（概念维度说明）

#### Step 2d：ACC 概念修正

```bash
# 自动概念修正 + 概念质量分析
python scripts/analyze_concepts.py \
    --dataset_name TOXICN \
    --concept_scores_dir data/processed/TOXICN/kipc/concept_scores \
    --mi_threshold 0.02 \
    --output_dir data/processed/TOXICN/kipc/concept_analysis
```

**输出文件**：
- `data/processed/TOXICN/kipc/concept_analysis/selected_concepts.json`（ACC 筛选后的概念索引）
- `data/processed/TOXICN/kipc/concept_analysis/concept_quality_report.txt`（概念质量报告）
- `data/processed/TOXICN/kipc/concept_analysis/concept_correlation.png`（概念-标签相关性图）

### 4.4 Phase 3：概念分类训练

```bash
# 训练直接线性概念分类器
python utils/concept_pipeline.py \
    --mode all \
    --dataset_name TOXICN \
    --concept_scores_dir data/processed/TOXICN/kipc/concept_scores \
    --selected_concepts data/processed/TOXICN/kipc/concept_analysis/selected_concepts.json \
    --epochs 500 \
    --batch_size 64 \
    --lr 1e-2 \
    --lambda_sparse 0.01 \
    --patience 20

# 仅测试概念分类器
python utils/concept_pipeline.py \
    --mode test \
    --timestamp <实验时间戳>
```

**输出文件**：
- `experiments/<timestamp>/config.json`
- `experiments/<timestamp>/best_model.pth`
- `experiments/<timestamp>/concept_weights.json`（概念权重，用于可解释性分析）
- `experiments/<timestamp>/metrics.png`
- `experiments/<timestamp>/test_results/`

### 4.5 Phase 4：迭代概念完备（可选）

```bash
# 如果概念模型 F1 与黑盒 F1 差距 > 0.01，运行迭代完备
# Step 1: 分析错误样本
python scripts/analyze_concepts.py \
    --dataset_name TOXICN \
    --concept_scores_dir data/processed/TOXICN/kipc/concept_scores \
    --error_analysis \
    --concept_timestamp <Phase3的时间戳> \
    --blackbox_timestamp <Phase1的时间戳>

# Step 2: 扩展 SAE 特征（增加隐层维度或 TopK）
python scripts/train_sae.py \
    --dataset_name TOXICN \
    --sae_hidden_dim 8192 \
    --topk 128 \
    --num_reserved 17 \
    --epochs 100

# Step 3: 重新计算概念评分 + ACC + 重新训练
# 重复 Phase 2c → 2d → Phase 3
```

### 4.6 完整一键运行（推荐按阶段手动运行）

```bash
# ========== Phase 1: 黑盒训练 ==========
python utils/blackbox_pipeline.py --mode all --dataset_name TOXICN --use_dual_encoder --use_coded_terms --use_homophone --use_multitask --epochs 30 --patience 10

# 验证：检查输出中 Val F1 是否 > 0.84
# 记录时间戳，例如 20260601-143000

# ========== Phase 2a: LLM 伪标签 ==========
python scripts/generate_pseudo_labels_vllm.py --mode train --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ --quantization awq
python scripts/generate_pseudo_labels_vllm.py --mode test --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ --quantization awq

# ========== Phase 2b: SAE 训练 ==========
python scripts/extract_representations.py --dataset_name TOXICN --blackbox_timestamp 20260601-143000 --mode all
python scripts/train_sae.py --dataset_name TOXICN --sae_hidden_dim 4096 --topk 64 --num_reserved 17
python scripts/name_sae_features_vllm.py --dataset_name TOXICN --model_name Qwen2.5-7B-Instruct-AWQ --quantization awq --sae_timestamp <SAE时间戳>

# ========== Phase 2c: 概念评分 ==========
python scripts/compute_concept_scores.py --dataset_name TOXICN --blackbox_timestamp 20260601-143000 --sae_timestamp <SAE时间戳> --mode all --use_structured_probe --use_sentence_embedding --use_sae_discovery

# ========== Phase 2d: ACC 概念修正 ==========
python scripts/analyze_concepts.py --dataset_name TOXICN --concept_scores_dir data/processed/TOXICN/kipc/concept_scores --mi_threshold 0.02

# ========== Phase 3: 概念分类 ==========
python utils/concept_pipeline.py --mode all --dataset_name TOXICN --epochs 500 --batch_size 64 --lr 1e-2 --lambda_sparse 0.01 --patience 20

# ========== Phase 4: 迭代完备（如果需要）==========
# 重复 Phase 2c-3，调整参数
```

### 4.7 依赖模型清单

| 模型 | 路径 | 用途 | 是否需要下载 |
|------|------|------|------------|
| chinese-roberta-wwm-ext | models/chinese-roberta-wwm-ext | 双路编码器 | 应已存在 |
| Qwen2.5-7B-Instruct-AWQ | models/Qwen2.5-7B-Instruct-AWQ | LLM 伪标签 + SAE 特征命名 | 应已存在 |
| bge-base-zh-v1.5 | models/bge-base-zh-v1.5 | 句子嵌入概念评分（A 路） | **需要下载** |

**bge-base-zh-v1.5 下载方式**：
```bash
# 方式一：从 HuggingFace 下载到 models/ 目录
git lfs install
git clone https://huggingface.co/BAAI/bge-base-zh-v1.5 models/bge-base-zh-v1.5

# 方式二：从 ModelScope 下载（国内更快）
pip install modelscope
modelscope download --model Xorbits/bge-base-zh-v1.5 --local_dir models/bge-base-zh-v1.5
```

---

## 五、训练策略详细说明

### Phase 1：黑盒训练（~30 分钟）

```
目标：F1 > 0.84
模型：知识增强双路 RoBERTa
损失：CE(toxic) + 0.3 * CE(topic) + 0.3 * CE(expression)
学习率：RoBERTa 2e-5, 分类头 1e-3
调度器：OneCycleLR
早停：patience=10，基于验证集 Macro F1
```

**关键验证点**：如果黑盒 F1 < 0.84，需要调整知识注入策略（增加编码术语、改进谐音还原等），不应进入 Phase 2。

### Phase 2：概念提取（~2 小时）

**Step 2a：LLM 伪标签生成**（离线，~1 小时）
- 用 Qwen2.5-7B-Instruct-AWQ 对训练集生成 intent 和 tone 分类伪标签
- 验证伪标签质量（与 TOXICN 标注的一致性、人工抽检）

**Step 2b：SAE 训练**（离线，~15 分钟）
- 用 Phase 1 的 RoBERTa 提取 h_fused 表示
- 训练 G-SAE，使用 TOXICN 标注作为条件化信号
- 用 Qwen2.5-7B 对 SAE 特征命名
- 筛选毒性相关特征

**Step 2c：概念评分计算**（离线，~10 分钟）
- S 路：训练线性探针，提取结构化概念得分
- A 路：用句子嵌入模型计算形容词概念得分
- D 路：用训练好的 SAE 提取发现概念得分

**Step 2d：ACC 概念修正**（~1 分钟）
- 计算每个概念与毒性标签的互信息
- 移除零/负相关概念
- 保留正相关的概念子集

### Phase 3：概念分类训练（~10 分钟）

```
模型：DirectConceptRegression（线性层 + L1 稀疏）
损失：CE(logits, labels) + λ_sparse * ||W||_1
学习率：1e-2
调度器：OneCycleLR
早停：patience=20，基于验证集 Macro F1
```

### Phase 4：迭代概念完备（可选，~30 分钟/轮）

```
while F1_blackbox - F1_concept > 0.01:
    1. 分析错误样本
    2. 增加 SAE 特征或生成新概念
    3. 重新计算概念评分 + ACC
    4. 重新训练概念分类器
```

### 可选：IB 约束增强

如果概念模型的性能接近黑盒但概念可解释性不足（通过概念干预测试验证），可以对 S 路概念探针施加 IB 约束：

```python
class IBConceptProbe(nn.Module):
    def __init__(self, input_dim=1536):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # [mu, log_var]
        )

    def forward(self, h_fused):
        params = self.encoder(h_fused)
        mu, log_var = params[:, 0], params[:, 1]
        std = torch.exp(0.5 * log_var)
        z = mu + torch.randn_like(std) * std
        concept_prob = torch.sigmoid(z)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return concept_prob, kl_loss
```

---

## 五、创新性论证

### 5.1 与现有工作的关系

| 现有工作 | KIPC-CBM 的继承 | KIPC-CBM 的超越 |
|---------|----------------|----------------|
| PCBM (ICLR 2024) | 事后概念提取范式 | 首次应用于 NLP 有害言论 + 三路概念提取 |
| Label-free CBM (ICLR 2024) | 零样本概念评分 | 句子嵌入替代 CLIP + 中文毒性适配 |
| CB-LLM (ICLR 2025) | ACC + 稀疏线性层 | 三路概念融合 + 迭代完备 |
| DCR (ECCV 2024) | 直接线性分类 | 首次用于 NLP 防概念绕过 |
| DN-CBM (ECCV 2024) | SAE 概念发现 | G-SAE 条件化 + 与结构化概念互补 |
| MCBM (ICLR 2026) | IB 约束 | 可选增强，不作为核心依赖 |
| CT-CBM (EMNLP 2025) | 迭代概念完备 | 与 PCBM 范式结合 |
| Concept2CLIP (CVPR 2024) | 概念泄漏防护 | 通过 ACC + 直接线性分类实现 |
| SCBM (IPM 2025) | 概念空间思想 | 完全重构：事后提取 + 三路概念 |

### 5.2 核心创新点

1. **首次将 PCBM 事后提取范式应用于 NLP 有害言论检测**
   - CV 中 PCBM 依赖 CLIP 的图像-文本对齐，NLP 中无此条件
   - 创新性地提出三路概念提取（结构化探测 + 句子嵌入 + SAE 发现）替代 CLIP
   - 根本性解决"训练中性能-可解释性博弈"问题

2. **首次将 Label-free CBM 的零样本概念评分迁移到 NLP**
   - 用句子嵌入相似度替代 LLM Likert 评分
   - 直接解决 LLM 对中文隐式毒性评分系统性偏低的问题
   - 更稳定、更高效、更连续

3. **首次在 NLP CBM 中引入 ACC 自动概念修正**
   - CB-LLM 证明 ACC 是性能匹配黑盒的关键
   - 之前所有方案都缺少这一步
   - 确保概念-标签对齐，去除噪声概念

4. **首次将 DCR 直接线性分类用于 NLP 防概念绕过**
   - 无隐藏层、无残差路径、L1 稀疏
   - 从根本上防止概念绕过，保证可解释性
   - 权重直接反映概念贡献

5. **三路概念融合 + 迭代完备**
   - S 路（结构化）：保证可解释性骨架
   - A 路（句子嵌入）：替代不可靠的 LLM Likert
   - D 路（SAE 发现）：保证概念充分性
   - 迭代完备：确保概念集覆盖黑盒模型的所有信息

### 5.3 与之前方案（v3 KE-MCBM / v4 SCREAM-Net）的关键区别

| 维度 | v3/v4（联合训练） | KIPC-CBM（事后提取） |
|------|-----------------|---------------------|
| 范式 | 概念层 + 分类器联合训练 | 先训练黑盒，再提取概念 |
| 概念来源 | SAE 发现 + 结构化概念 | 三路提取（探测 + 嵌入 + SAE） |
| 概念评分 | LLM Likert（不可靠） | 句子嵌入相似度（更稳定） |
| 分类层 | 稀疏线性层 | 直接线性分类（DCR，更强防绕过） |
| 防绕过 | IB 约束（训练中可能被绕过） | 无残差 + 线性分类 + ACC（结构性防绕过） |
| 概念完备性 | 概念充分性验证（事后检查） | 迭代概念完备（主动补全） |
| 复杂度 | 5+ 模块（G-SAE + 推理图 + MCBL + 侧通道 + 稀疏层） | 3 模块（黑盒 + 概念提取 + 线性分类） |
| 训练稳定性 | 多损失项博弈，可能不稳定 | 分阶段训练，每阶段目标明确 |

---

## 六、实验设计

### 6.1 核心对比实验

| 实验组 | 配置 | 目的 |
|--------|------|------|
| Baseline-1 | 纯 RoBERTa 微调 | 天花板参考 (~0.83) |
| Baseline-2 | 知识增强 RoBERTa（无概念层） | 验证知识注入增益 |
| **KIPC-CBM (完整)** | **三路概念 + ACC + 直接线性分类** | **主实验** |
| KIPC-CBM w/o A 路 | 去掉句子嵌入概念 | 验证句子嵌入概念的增益 |
| KIPC-CBM w/o D 路 | 去掉 SAE 发现概念 | 验证 SAE 概念的增益 |
| KIPC-CBM w/o ACC | 去掉自动概念修正 | 验证 ACC 的必要性 |
| KIPC-CBM w/ MLP | 用 MLP 替代直接线性分类 | 验证 DCR 的必要性 |
| KIPC-CBM w/ Likert | 用 LLM Likert 替代句子嵌入 | 验证句子嵌入 vs Likert |
| KIPC-CBM w/ Residual | 加回残差路径 | 验证残差导致概念架空 |
| KI-SCB 复现 | v2 方案 | 前方案对比 |

### 6.2 关键超参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| SAE 隐层维度 | 4096 | 过完备 2.67× 扩展 |
| SAE TopK | 64 | 每样本最多激活 64 特征 |
| D 组概念数量 | 32-64 | 从 SAE 特征中筛选 |
| λ_sparse | 0.01 | L1 稀疏正则权重 |
| λ_IB（可选） | 0.01 → 0.1 | IB 约束权重（退火） |
| ACC 互信息阈值 | 0.02 | 概念-标签相关性筛选阈值 |
| 迭代完备 δ | 0.01 | 概念模型与黑盒的允许性能差距 |
| 句子嵌入模型 | BAAI/bge-base-zh-v1.5 | 中文句子嵌入 |

### 6.3 可解释性评估

| 评估维度 | 方法 |
|---------|------|
| 概念预测准确率 | S 路概念与 TOXICN 标注的一致性 |
| 概念干预有效性 | 修改某概念后分类 logit 变化是否符合预期 |
| 稀疏性 | 每个预测依赖的平均概念数量 |
| 权重可解释性 | W[j, y] 是否与概念 j 的语义一致 |
| 概念完备性 | 概念模型与黑盒模型的性能差距 |

### 6.4 预期结果

| 数据集 | 黑盒目标 | 概念模型目标 |
|--------|---------|------------|
| TOXICN | F1 ≥ 0.84 | F1 ≥ 0.84（匹配黑盒） |
| COLD | F1 ≥ 0.83 | F1 ≥ 0.82 |
| SWSR | F1 ≥ 0.80 | F1 ≥ 0.79 |

---

## 七、风险分析与缓解

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| 黑盒 F1 < 0.84 | 中 | 无法进入 Phase 2 | 调整知识注入策略；增加编码术语；尝试 FGM 对抗训练 |
| 句子嵌入对隐式毒性也不敏感 | 中 | A 路概念质量差 | 依赖 S 路 + D 路补偿；对还原文本计算嵌入 |
| 概念模型性能远低于黑盒 | 中 | F1 < 0.84 | 迭代概念完备；增加 SAE 特征；尝试非线性探针 |
| SAE 发现特征不可解释 | 中 | 可解释性受损 | LLM 命名 + 人工验证；仅保留高可解释性特征 |
| LLM 伪标签质量差 | 低 | S 路概念不准 | 用 TOXICN 标注验证；低质量维度降权 |
| 线性分类表达力不足 | 低 | F1 下降 | 迭代概念完备增加概念数；最后手段：加一层隐藏层 |

---

## 八、资源可行性（NVIDIA 3080Ti 12GB）

| 组件 | 参数量 | 显存 (fp16) |
|------|--------|------------|
| RoBERTa-wwm-ext (共享权重) | 102M | ~400MB |
| 双路输入（共享权重） | 0 | ~200MB |
| 知识预处理层 | ~100K | 可忽略 |
| 多任务分类头 | ~10K | 可忽略 |
| SAE（离线训练） | ~12M | 离线 |
| 线性探针（S 路） | ~26K | 可忽略 |
| 句子嵌入模型（离线） | ~100M | 离线 |
| 直接线性分类层 | ~150×2=300 | 可忽略 |
| **总计（主训练）** | **~102M** | **~3GB** |

**结论**：3080Ti 完全可行，batch_size=16 无压力。

---

## 九、实施路线图

### Phase 1：黑盒训练（1-2 天）
1. 实现知识增强双路 RoBERTa
2. 构建编码术语词表和谐音映射表
3. 训练黑盒分类器
4. 验证 F1 > 0.84

### Phase 2：概念提取（1-2 天）
1. LLM 伪标签生成（intent, tone）
2. SAE 训练 + 特征命名 + 筛选
3. 句子嵌入概念评分
4. 结构化概念探针训练
5. ACC 概念修正

### Phase 3：概念分类训练（1 天）
1. 实现直接线性分类
2. 训练 + 评估
3. 与黑盒性能对比

### Phase 4：迭代完备（1-2 天）
1. 错误分析
2. 概念补全
3. 重新训练
4. 直到性能匹配黑盒

### Phase 5：评估与分析（1-2 天）
1. 核心对比实验
2. 消融实验
3. 可解释性评估
4. 跨数据集泛化

---

## 十、总结

### 10.1 为什么这次可能成功

之前所有方案的失败可以归结为一个根本问题：**在训练过程中同时优化性能和可解释性，模型总是选择性能**。无论添加什么约束（IB、推理图、门控），只要存在绕过的可能（残差路径、非线性分类器），模型就会利用。

KIPC-CBM 通过**范式转变**解决这个问题：

1. **事后提取范式**：先保证性能（黑盒 > 0.84），再提取可解释性。不存在训练中的博弈。
2. **直接线性分类**：无残差路径、无隐藏层，从结构上杜绝绕过的可能。
3. **句子嵌入替代 Likert**：直接解决 LLM 对中文隐式毒性评分偏低的问题。
4. **ACC 概念修正**：确保概念-标签对齐，去除噪声概念。
5. **迭代概念完备**：主动补全概念直到性能匹配黑盒。

### 10.2 最坏情况下的保底方案

如果概念模型性能仍低于黑盒 >1%：
1. **PCBM-h 式残差**：添加正则化残差路径，但限制其 L2 范数不超过概念路径
2. **非线性概念探针**：将线性探针升级为浅层 MLP（1 层隐藏层）
3. **概念交互特征**：添加概念间的交互项（如 "地域 × 反讽"）

这些保底方案会牺牲部分可解释性，但保证性能达标。
