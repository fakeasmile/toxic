from pathlib import Path


class CGRMConfig():
    def __init__(self):
        # ========== 路径相关配置 ==========
        self.base_path = Path(__file__).parent.parent
        self.raw_data_path = self.base_path / "data" / "raw"
        self.processed_path = self.base_path / "data" / "processed"
        self.models_path = self.base_path / "models"
        self.experiment_path = self.base_path / "experiments"

        # ========== 数据集配置 ==========
        self.dataset_name = "TOXICN"
        self.plm_name = "chinese-roberta-wwm-ext"
        self.plm_local_path = str(self.models_path / self.plm_name)
        self.max_length = 128

        # ========== 随机种子配置 ==========
        self.seed = 1
        self.use_deterministic = False

        # ========== CGRM 模型参数 ==========
        self.num_concepts = 56
        self.concept_dim = 64
        self.hidden_dim = 768
        self.num_heads = 4
        self.dropout_rate = 0.3

        # ========== CGRM 训练超参数 ==========
        self.batch_size = 16
        self.epochs = 30
        self.plm_lr = 2e-5
        self.max_lr = 1e-4
        self.pct_start = 0.1
        self.div_factor = 25.0
        self.final_div_factor = 10000.0
        self.anneal_strategy = 'cos'
        self.patience = 5

    def __repr__(self):
        lines = [f"{self.__class__.__name__}:"]
        for key, value in sorted(self.__dict__.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


if __name__ == '__main__':
    config = CGRMConfig()
    print(config)
