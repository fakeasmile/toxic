"""ICB-CBM配置类"""

from pathlib import Path


class ICBConfig:
    """ICB-CBM训练配置

    继承MLPConfig的基本路径和种子配置，新增ICB-CBM特有的参数。
    """

    def __init__(self):
        # ========== 路径相关配置 ==========
        self.base_path = Path(__file__).parent.parent
        self.raw_data_path = self.base_path / "data" / "raw"
        self.processed_path = self.base_path / "data" / "processed"
        self.adjective_path = self.raw_data_path / "adjective" / "toxic_adjectives_new.csv"
        self.models_path = self.base_path / "models"
        self.experiment_path = self.base_path / "experiments"

        # ========== 随机种子配置 ==========
        self.seed = 1
        self.use_deterministic = False

        # ========== 数据集配置 ==========
        self.dataset_name = "TOXICN"
        self.model_name = "Qwen2.5-7B-Instruct"

        # ========== ICB-CBM模型参数 ==========
        self.dense_dim = 64          # 每个概念的稠密向量维度 d
        self.num_residual = 32       # 残差概念数量 K_r
        self.hidden_features = 128   # 分类器隐藏层维度
        self.dropout_rate = 0.3      # Dropout比率
        self.layer = 16              # LLM提取hidden state的层号

        # ========== ICB-CBM损失权重 ==========
        self.alpha = 0.01            # IB损失权重
        self.gamma = 0.001           # 稀疏损失权重

        # ========== ICC概念选择 ==========
        self.icc_threshold = 0.05    # ICC概念选择相关性阈值

        # ========== 训练超参数 ==========
        self.batch_size = 32
        self.epochs = 200

        # OneCycleLR 学习率调度器参数
        self.max_lr = 1e-3
        self.pct_start = 0.2
        self.div_factor = 25.0
        self.final_div_factor = 10000.0
        self.anneal_strategy = 'cos'

        self.patience = 30           # 早停耐心值

    def __repr__(self):
        """返回配置对象的字符串表示"""
        lines = [f"{self.__class__.__name__}:"]
        for key, value in sorted(self.__dict__.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


if __name__ == '__main__':
    config = ICBConfig()
    print(config)
