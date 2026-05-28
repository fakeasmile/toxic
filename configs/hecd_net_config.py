from pathlib import Path


class HECDNetConfig:
    def __init__(self):
        # ========== 路径配置 ==========
        self.base_path = Path(__file__).parent.parent
        self.raw_data_path = self.base_path / "data" / "raw"
        self.processed_path = self.base_path / "data" / "processed"
        self.experiment_path = self.base_path / "experiments"
        self.models_path = self.base_path / "models"

        self.concept_csv_path = self.raw_data_path / "adjective" / "discovered_concepts.csv"
        self.concept_graph_path = self.processed_path / "concept_graph.json"
        self.dataset_name = "TOXICN"
        self.plm_name = "chinese-roberta-wwm-ext"
        # 本地模型路径（优先使用本地，避免连接 HuggingFace）
        self.plm_local_path = str(self.models_path / self.plm_name)

        # ========== 数据配置 ==========
        self.max_length = 128
        self.num_concepts = 56
        self.num_classes = 2
        self.num_topics = 4       # race, gender, region, lgbt
        self.num_expressions = 4  # 0,1,2,3
        self.num_targets = 5      # multi-label
        self.num_platforms = 2    # zhihu, tieba

        # ========== 模型配置 ==========
        self.plm_hidden_size = 768
        self.concept_dim = 64
        self.num_gat_layers = 2
        self.gat_hidden_dim = 128
        self.gat_num_heads = 4
        self.adapter_dim = 64
        self.dropout = 0.3
        self.use_residual = True
        self.use_graph = True
        self.use_adapter = True
        self.use_contrastive = True
        self.use_auxiliary = True

        # ========== 训练配置 ==========
        self.seed = 1
        self.use_deterministic = False
        self.batch_size = 16
        self.epochs = 30
        self.lr = 1e-4
        self.plm_lr = 2e-5
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.patience = 5
        self.gradient_accumulation_steps = 1

        # ========== 损失权重 ==========
        self.aux_weight = 0.3
        self.contrastive_weight = 0.1
        self.graph_reg_weight = 0.01
        self.contrastive_temperature = 0.1

        # ========== 设备配置 ==========
        self.device = "cuda"
        self.fp16 = True

    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, Path):
                d[k] = str(v)
            else:
                d[k] = v
        return d

    def __repr__(self):
        lines = [f"{self.__class__.__name__}:"]
        for key, value in sorted(self.to_dict().items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


if __name__ == "__main__":
    cfg = HECDNetConfig()
    print(cfg)
