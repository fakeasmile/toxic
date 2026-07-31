import torch
from torch import nn


# 概念类型常量
THREE_LEVEL_TYPES = {"evaluation", "discrimination", "neutral"}
BINARY_TYPES = {"behavior", "strategy", "intent", "effect"}


class MLP(nn.Module):
    def __init__(self, in_features, dropout_rate=0.5, hidden_features=96):
        """
        概念瓶颈分类器，使用矩阵门控。

        结构: 门控(sigmoid(W@x)) → Dropout → FC(in→hidden) → ReLU → Dropout → FC(hidden→2)

        Args:
            in_features: 输入特征维度（概念向量维度）
            dropout_rate: dropout 比率
            hidden_features: 隐藏层维度
        """
        super(MLP, self).__init__()

        # ========== 门控单元 ==========
        # 矩阵门控：gate = sigmoid(W @ x)，样本相关
        self.gate_layer = nn.Linear(in_features, in_features)

        # ========== dropout ==========
        self.dropout = nn.Dropout(dropout_rate)

        # ========== 分类层 ==========
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()

    def forward(self, feature_vector):
        # 门控
        gate_weights = torch.sigmoid(self.gate_layer(feature_vector))
        x = feature_vector * gate_weights

        # dropout
        x = self.dropout(x)

        # 分类
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class FormConditionedMLP(nn.Module):
    """Form-Conditioned Gate概念瓶颈分类器。

    核心思路：文本形式特征(form)不参与分类，而是调节语义概念的门控。
    gate = sigmoid(W_sem @ sem + W_form @ form)
    form作为"旁证"修正语义gate的倾向，分类路径仍为177维概念。

    Args:
        in_features: 语义概念向量维度（177）
        form_dim: 文本形式特征维度（10）
        dropout_rate: dropout比率
        hidden_features: 隐藏层维度
    """

    def __init__(self, in_features, form_dim, dropout_rate=0.5, hidden_features=96):
        super(FormConditionedMLP, self).__init__()

        # ========== 门控单元 ==========
        # 语义自门控 + form条件化调整
        # form_gate_layer无bias：让gate_layer的bias统一承担偏置，避免冗余
        self.gate_layer = nn.Linear(in_features, in_features)
        self.form_gate_layer = nn.Linear(form_dim, in_features, bias=False)

        # ========== dropout ==========
        self.dropout = nn.Dropout(dropout_rate)

        # ========== 分类层 ==========
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()

    def forward(self, sem_vector, form_vector):
        # 门控：sem自门控 + form条件化调整
        gate_weights = torch.sigmoid(
            self.gate_layer(sem_vector) + self.form_gate_layer(form_vector)
        )
        x = sem_vector * gate_weights

        # dropout
        x = self.dropout(x)

        # 分类
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_gate_values(self, sem_vector, form_vector):
        """返回门控值，用于诊断分析。

        Returns:
            gate_total: sigmoid(W_sem @ sem + W_form @ form) 完整门控
            gate_sem_only: sigmoid(W_sem @ sem) 纯语义门控
            gate_form_raw: W_form @ form form对门控的原始贡献（sigmoid前）
        """
        with torch.no_grad():
            gate_sem = self.gate_layer(sem_vector)
            gate_form = self.form_gate_layer(form_vector)
            gate_total = torch.sigmoid(gate_sem + gate_form)
            gate_sem_only = torch.sigmoid(gate_sem)
        return gate_total, gate_sem_only, gate_form


class SimpleMLP(nn.Module):
    """无门控的简单MLP分类器。

    结构: FC(in→hidden) → ReLU → Dropout → FC(hidden→2)

    Args:
        in_features: 输入特征维度
        dropout_rate: dropout比率
        hidden_features: 隐藏层维度
    """

    def __init__(self, in_features, dropout_rate=0.5, hidden_features=96):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class HierarchicalTypeGatedMLP(nn.Module):
    """层次化类型门控MLP，利用概念类型的结构化关系。

    核心思路：概念按语义角色分组，因果链方向的跨类型条件化门控。
    - Pragmatic组(intent+effect): 因果链起点，自门控
    - Action组(behavior+strategy): Pragmatic条件化门控
    - Target组(evaluation+discrimination): Action条件化门控
    - Surface组(neutral): 自门控（正交特征）

    因果链: intent(为什么) → behavior(做什么) → effect(什么影响)
                    → strategy(怎么做) →
            behavior(做什么) → evaluation(什么性质)
                             → discrimination(针对谁)

    Args:
        in_features: 输入特征维度（134 for single mode）
        concept_types: 概念类型列表，如 ["behavior","behavior",...,"effect"]
        dropout_rate: dropout比率
        hidden_features: 隐藏层维度
    """

    # 类型分组定义：组名 → (包含的prompt_template类型, 因果上游组名)
    GROUP_DEFS = {
        "pragmatic": {"types": {"intent", "effect"}, "conditioned_by": None},
        "action":    {"types": {"behavior", "strategy"}, "conditioned_by": "pragmatic"},
        "target":    {"types": {"evaluation", "discrimination"}, "conditioned_by": "action"},
        "surface":   {"types": {"neutral"}, "conditioned_by": None},
    }

    def __init__(self, in_features, concept_types, dropout_rate=0.5, hidden_features=96):
        super(HierarchicalTypeGatedMLP, self).__init__()

        # 根据concept_types计算每个组的索引和维度
        self.group_indices = {}
        self.group_dims = {}
        for group_name, group_def in self.GROUP_DEFS.items():
            indices = [i for i, t in enumerate(concept_types) if t in group_def["types"]]
            self.group_indices[group_name] = indices
            self.group_dims[group_name] = len(indices)

        # 自门控层：每组一个
        self.self_gates = nn.ModuleDict()
        for name, dim in self.group_dims.items():
            if dim > 0:
                self.self_gates[name] = nn.Linear(dim, dim)

        # 跨类型条件化层：上游组 → 下游组（无bias，让自门控的bias统一承担偏置）
        self.cross_gates = nn.ModuleDict()
        for group_name, group_def in self.GROUP_DEFS.items():
            source = group_def["conditioned_by"]
            if source is not None and self.group_dims[group_name] > 0:
                key = f"{source}_to_{group_name}"
                self.cross_gates[key] = nn.Linear(
                    self.group_dims[source], self.group_dims[group_name], bias=False
                )

        # 分类层
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 分组提取
        group_features = {}
        for name, indices in self.group_indices.items():
            if indices:
                group_features[name] = x[:, indices]

        # 层次化门控（按因果链顺序）
        gated_features = {}

        for group_name in ["pragmatic", "action", "target", "surface"]:
            feat = group_features.get(group_name)
            if feat is None:
                continue

            # 自门控
            gate = self.self_gates[group_name](feat)

            # 跨类型条件化
            source = self.GROUP_DEFS[group_name]["conditioned_by"]
            if source is not None and source in gated_features:
                key = f"{source}_to_{group_name}"
                gate = gate + self.cross_gates[key](gated_features[source])

            gate_weights = torch.sigmoid(gate)
            gated_features[group_name] = feat * gate_weights

        # 重组：按原始概念顺序
        x_gated = torch.zeros_like(x)
        for name, indices in self.group_indices.items():
            if indices and name in gated_features:
                x_gated[:, indices] = gated_features[name]

        # 分类
        x = self.dropout(x_gated)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class TypeAugmentedGatedMLP(nn.Module):
    """类型增强门控MLP：保留全矩阵门控 + 类型级汇总特征。

    在标准GatedMLP基础上，为分类层添加每个概念类型的统计汇总特征
    (mean, max)，使模型在细粒度概念分数之外，还能利用粗粒度的
    类型级信息。这为分类器提供了"鸟瞰"视角。

    结构:
      gate = sigmoid(W @ x)       ← 全矩阵门控（同GatedMLP）
      x_gated = x * gate
      summary = [mean_t(x_gated), max_t(x_gated) for t in types]  ← 7类型×2统计=14维
      h = FC(concat[x_gated, summary] → hidden) → ReLU → Dropout → FC(2)

    Args:
        in_features: 输入特征维度
        concept_types: 概念类型列表
        dropout_rate: dropout比率
        hidden_features: 隐藏层维度
    """

    def __init__(self, in_features, concept_types, dropout_rate=0.5, hidden_features=96):
        super(TypeAugmentedGatedMLP, self).__init__()

        # 记录每种类型的索引
        self.type_names = sorted(set(concept_types))
        self.type_indices = {}
        for t in self.type_names:
            self.type_indices[t] = [i for i, ct in enumerate(concept_types) if ct == t]

        # 类型级汇总特征维度: 7类型 × 2统计(mean, max)
        n_summary = len(self.type_names) * 2

        # 全矩阵门控（同GatedMLP）
        self.gate_layer = nn.Linear(in_features, in_features)

        # 分类层（输入维度 = 概念维度 + 汇总维度）
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(in_features + n_summary, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 全矩阵门控（同GatedMLP）
        gate_weights = torch.sigmoid(self.gate_layer(x))
        x_gated = x * gate_weights

        # 类型级汇总特征
        summary_parts = []
        for t in self.type_names:
            idx = self.type_indices[t]
            type_feats = x_gated[:, idx]
            summary_parts.append(type_feats.mean(dim=1, keepdim=True))
            summary_parts.append(type_feats.max(dim=1, keepdim=True)[0])
        summary = torch.cat(summary_parts, dim=1)

        # 拼接门控特征 + 汇总特征
        x = torch.cat([x_gated, summary], dim=1)

        # 分类
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class DivergenceAwareGatedMLP(nn.Module):
    """分歧感知门控MLP：全矩阵门控 + 类型汇总 + 跨类型分歧特征。

    核心创新：显式建模概念类型之间的分歧模式，捕捉扁平分类器无法利用的结构化信号。

    关键分歧模式（来自失败样本分析）：
    - discrimination > evaluation: 隐含歧视（高群体针对性但低危害评估）→ FN漏检修正
    - neutral > evaluation: 隐含有害（高表面特征但低危害评估）→ FN漏检修正
    - intent > behavior: 意图-行为断裂（高意图但低行为实施）→ 隐含攻击信号
    - effect > behavior: 效果-行为断裂（高影响但低直接行为）→ 间接伤害信号
    - evaluation P(2)高: "讨论但无害"的anti-pattern → FP误判修正

    结构:
      1. gate = sigmoid(W @ x)               ← 全矩阵门控（同GatedMLP）
      2. x_gated = x * gate
      3. type_summary = [mean_t, max_t for t in types]     ← 7类型×2=14维
      4. divergence = [mean_src - mean_tgt, max_src - max_tgt for (src,tgt) in pairs] ← 5对×2=10维
      5. p2_anti_pattern = mean[P(2)/(P(2)+P(3)) for 3-level eval+neutral]  ← 1维
      6. x = concat[x_gated, type_summary, divergence, p2_anti]  ← 189+14+10+1=214维
      7. FC(214, 96) → ReLU → Dropout → FC(96, 2)

    Args:
        in_features: 输入特征维度（conditional模式=189）
        concept_types: 概念类型列表（134个值）
        dropout_rate: dropout比率
        hidden_features: 隐藏层维度
    """

    # 跨类型分歧对：(源类型, 目标类型, 语义含义)
    DIVERGENCE_PAIRS = [
        ("discrimination", "evaluation"),   # 隐含歧视：针对群体但未被评估为有害
        ("neutral", "evaluation"),          # 隐含有害：有表面特征但未被评估为有害
        ("intent", "behavior"),             # 意图-行为断裂：有攻击意图但未直接实施
        ("effect", "behavior"),             # 效果-行为断裂：有负面影响但无直接行为
        ("discrimination", "neutral"),      # 歧视-表面分歧：针对群体但不只是表面特征
    ]

    def __init__(self, in_features, concept_types, dropout_rate=0.5, hidden_features=96):
        super(DivergenceAwareGatedMLP, self).__init__()

        # 类型索引映射
        self.type_names = sorted(set(concept_types))
        self.type_indices = {}
        for t in self.type_names:
            self.type_indices[t] = [i for i, ct in enumerate(concept_types) if ct == t]

        # 3级概念中evaluation和neutral的索引（用于P(2) anti-pattern计算）
        self.eval_neutral_indices = []
        for i, ct in enumerate(concept_types):
            if ct in ("evaluation", "neutral"):
                self.eval_neutral_indices.append(i)

        # 特征维度计算
        n_type_summary = len(self.type_names) * 2  # 7 types × (mean, max) = 14
        n_divergence = len(self.DIVERGENCE_PAIRS) * 2  # 5 pairs × (mean_diff, max_diff) = 10
        n_p2_anti = 1  # P(2) anti-pattern signal for evaluation+neutral

        # 全矩阵门控（同GatedMLP）
        self.gate_layer = nn.Linear(in_features, in_features)

        # 分类层
        total_features = in_features + n_type_summary + n_divergence + n_p2_anti
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(total_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()

    def forward(self, x, concept_scores):
        """前向传播。

        Args:
            x: 主特征向量 (batch, in_features)，来自concept_features.py
            concept_scores: 概念标量分数 (batch, 134)，P(2) for binary / P(3) for 3-level
        """
        # 1. 全矩阵门控
        gate_weights = torch.sigmoid(self.gate_layer(x))
        x_gated = x * gate_weights

        # 2. 类型汇总特征（从concept_scores计算）
        type_means = []
        type_maxes = []
        for t in self.type_names:
            idx = self.type_indices[t]
            type_feats = concept_scores[:, idx]
            type_means.append(type_feats.mean(dim=1, keepdim=True))
            type_maxes.append(type_feats.max(dim=1, keepdim=True)[0])

        # 3. 分歧特征（从concept_scores计算）
        divergence_feats = []
        for src_type, tgt_type in self.DIVERGENCE_PAIRS:
            src_idx = self.type_indices[src_type]
            tgt_idx = self.type_indices[tgt_type]
            src_feats = concept_scores[:, src_idx]
            tgt_feats = concept_scores[:, tgt_idx]

            # mean差异
            src_mean = src_feats.mean(dim=1, keepdim=True)
            tgt_mean = tgt_feats.mean(dim=1, keepdim=True)
            divergence_feats.append(src_mean - tgt_mean)

            # max差异
            src_max = src_feats.max(dim=1, keepdim=True)[0]
            tgt_max = tgt_feats.max(dim=1, keepdim=True)[0]
            divergence_feats.append(src_max - tgt_max)

        # 4. P(2) anti-pattern信号（evaluation+neutral的"存在但无害"比例）
        # 高P(2)/(P(2)+P(3)) 意味着"讨论了但无害"，是FP的修正信号
        # 注意：concept_scores中evaluation/neutral存的是P(3)，我们需要从level_probs获取P(2)
        # 这里使用一个近似：evaluation+neutral类型的concept_scores越低，
        # 相对而言P(2)占比越高（因为P(1)+P(2)+P(3)=1，P(3)低→P(1)+P(2)高）
        # 直接用1 - mean(eval+neutral scores)作为anti-pattern proxy
        if self.eval_neutral_indices:
            en_scores = concept_scores[:, self.eval_neutral_indices]
            p2_anti = 1.0 - en_scores.mean(dim=1, keepdim=True)
        else:
            p2_anti = torch.zeros(x.size(0), 1, device=x.device)

        # 5. 拼接所有特征
        summary = torch.cat(type_means + type_maxes, dim=1)
        divergence = torch.cat(divergence_feats, dim=1)
        x = torch.cat([x_gated, summary, divergence, p2_anti], dim=1)

        # 6. 分类
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class FormConditionedSimpleMLP(nn.Module):
    """Form-Conditioned无门控MLP。

    核心思路：form特征作为隐藏层的偏置调整，条件化语义特征的解读。
    h = ReLU(W @ sem + W_form @ form + b)
    form只影响隐藏层激活阈值，不改变分类路径的输入维度。

    Args:
        in_features: 语义概念向量维度（177）
        form_dim: 文本形式特征维度（10）
        dropout_rate: dropout比率
        hidden_features: 隐藏层维度
    """

    def __init__(self, in_features, form_dim, dropout_rate=0.5, hidden_features=96):
        super(FormConditionedSimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.form_bias = nn.Linear(form_dim, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sem_vector, form_vector):
        # form特征调整隐藏层偏置，条件化语义解读
        x = self.fc1(sem_vector) + self.form_bias(form_vector)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
