from pathlib import Path


class PCCGConfig:
    """PCCG (Pragmatic Causal Concept Graph) 配置"""

    def __init__(self):
        # ========== 路径相关配置 ==========
        self.base_path = Path(__file__).parent.parent
        self.raw_data_path = self.base_path / "data" / "raw"
        self.processed_path = self.base_path / "data" / "processed"
        self.adjective_path = self.raw_data_path / "adjective" / "pragmatic_concepts.csv"
        self.models_path = self.base_path / "models"
        self.experiment_path = self.base_path / "experiments"

        # ========== 随机种子配置 ==========
        self.seed = 1
        self.use_deterministic = False

        # ========== 数据集配置 ==========
        self.dataset_name = "TOXICN"
        self.model_name = "Qwen2.5-7B-Instruct-AWQ"

        # ========== 因果图配置 ==========
        # 7个语用推理维度（顺序与pragmatic_concepts.csv的category列顺序一致）
        self.node_names = [
            "expression_strategy",    # 表达策略
            "implicit_intent",        # 隐含意图
            "encoding_strategy",      # 编码策略
            "attack_target",          # 攻击目标
            "emotional_tone",         # 情感基调
            "pragmatic_effect",       # 语用效果
            "topic_distinction",      # 话题区分
        ]

        # 因果边 (src_idx, dst_idx)，定义维度间的因果关系
        # 节点索引: 0=表达策略, 1=隐含意图, 2=编码策略, 3=攻击目标, 4=情感基调, 5=语用效果, 6=话题区分
        self.causal_edges = [
            (6, 0),  # 话题区分 → 表达策略
            (0, 2),  # 表达策略 → 编码策略
            (0, 3),  # 表达策略 → 攻击目标
            (2, 1),  # 编码策略 → 隐含意图
            (1, 3),  # 隐含意图 → 攻击目标
            (1, 4),  # 隐含意图 → 情感基调
            (3, 4),  # 攻击目标 → 情感基调
            (6, 4),  # 话题区分 → 情感基调
            (4, 5),  # 情感基调 → 语用效果
        ]
        # 自环：确保无入边节点也能参与GAT消息传递
        self.causal_edges_with_self = self.causal_edges + [
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)
        ]

        # 每个维度内的Likert概念数量（与pragmatic_concepts.csv的category列严格对应）
        # CSV实际统计: 表达策略13, 隐含意图14, 编码策略7, 攻击目标26, 情感基调10, 语用效果10, 话题区分7, 正面概念4
        self.dim_concept_counts = {
            "expression_strategy": 13,
            "implicit_intent": 14,
            "encoding_strategy": 7,
            "attack_target": 26,
            "emotional_tone": 10,
            "pragmatic_effect": 10,
            "topic_distinction": 7,
            # 正面概念4个不归入任何因果图节点，直接拼接到Likert向量末尾
        }
        self.positive_concept_count = 4  # 正面概念数量
        self.num_likert_concepts = sum(self.dim_concept_counts.values()) + self.positive_concept_count  # 87 + 4 = 91

        # ========== BGE嵌入维度 ==========
        self.bge_dim = 768

        # ========== GNN配置 ==========
        self.gnn_hidden_dim = 256
        self.gnn_num_heads = 4
        self.gnn_num_layers = 2

        # ========== 信息瓶颈配置 ==========
        self.ib_beta_min = 0.01       # IB初始强度
        self.ib_beta_target = 1.0     # IB目标强度
        self.ib_warmup_ratio = 0.2    # IB退火warmup比例

        # ========== 分类器配置 ==========
        self.hidden_features = 128
        self.dropout_rate = 0.3

        # ========== 训练超参数 ==========
        self.batch_size = 32
        self.epochs = 200
        self.patience = 30

        # OneCycleLR
        self.max_lr = 1e-3
        self.pct_start = 0.2
        self.div_factor = 25.0
        self.final_div_factor = 10000.0
        self.anneal_strategy = 'cos'

        # ========== 损失权重 ==========
        self.alpha_ib = 0.01          # IB损失权重
        self.gamma_sparse = 0.001     # 稀疏门控L1权重
        self.gamma_consist = 0.1      # 一致性约束权重

    def __repr__(self):
        lines = [f"{self.__class__.__name__}:"]
        for key, value in sorted(self.__dict__.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


if __name__ == '__main__':
    config = PCCGConfig()
    print(config)
