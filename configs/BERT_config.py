from pathlib import Path


class BERTConfig:
    def __init__(self):
        # ========== 路径相关配置 ==========
        self.base_path = Path(__file__).parent.parent  # 项目根目录

        self.dataset_name = "TOXICN"  # 数据集名称（TOXICN/COLD）
        self.train_path = self.base_path / "data" / "raw" / self.dataset_name / "train.json"  # 训练集路径
        self.test_path = self.base_path / "data" / "raw" / self.dataset_name / "test.json"  # 测试集路径
        self.bert_path = self.base_path / "models" / "bert-base-chinese"  # bert-base-chinese 模型路径
        self.experiment_path = self.base_path / "experiments_bert"  # 实验结果保存目录

        # ========== 随机种子配置 ==========
        self.seed = 1  # 随机种子
        self.use_deterministic = False  # 启用确定性算法以确保可复现

        # ========== BERT 训练超参数 ==========
        self.batch_size = 16  # 批次大小
        self.epochs = 5  # 训练轮数
        self.learning_rate = 2e-5  # 学习率
        self.warmup_ratio = 0.1  # Warmup 步数占总训练步数的比例
        self.weight_decay = 0.01  # 权重衰减（L2正则化）
        self.max_seq_length = 128  # 最大序列长度

        # ========== BERT 模型参数 ==========
        self.dropout_rate = 0.1  # Dropout 比率
        self.patience = 2  # 早停耐心值（验证集F1连续patience个epoch未提升则停止训练）
