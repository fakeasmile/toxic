from pathlib import Path


class DualChannelFusionConfig():
    def __init__(self):
        # ========== 路径相关配置 ==========
        self.base_path = Path(__file__).parent.parent  # 基目录
        self.raw_data_path = self.base_path / "data" / "raw"  # 原始数据根目录
        self.processed_path = self.base_path / "data" / "processed"  # 数据处理保存路径
        self.bert_path = self.base_path / "models" / "bert-base-chinese"  # bert-base-chinese 模型路径
        self.experiment_path = self.base_path / "experiments_dual_channel_fusion"  # 实验结果保存目录

        # ========== 随机种子配置 ==========
        self.seed = 1  # 随机种子
        self.use_deterministic = False  # 启用确定性算法以确保可复现

        # ========== BERT 训练超参数 ==========
        self.batch_size = 16  # 批次大小
        self.epochs = 5  # 训练轮数 (用户要求不可修改)
        self.learning_rate = 2e-5  # 基础学习率
        self.bert_learning_rate = 1e-5  # BERT层学习率（更低，保护预训练知识）
        self.projection_learning_rate = 1e-4  # 投影层学习率（更高，快速适应）
        self.warmup_ratio = 0.1  # Warmup 占总训练步数的比例
        self.weight_decay = 0.01  # 权重衰减（L2正则化）
        self.max_seq_length = 128  # 最大序列长度

        # ========== 融合模型参数 ==========
        self.dropout_rate = 0.3  # Dropout 比率
        self.patience = 2  # 早停耐心值
        self.label_smoothing = 0.1  # 标签平滑系数
        self.use_focal_loss = True  # 是否使用Focal Loss
        self.focal_gamma = 2.0  # Focal Loss gamma参数

    def __repr__(self):
        """返回配置对象的字符串表示，包含所有配置项"""
        lines = [f"{self.__class__.__name__}:"]
        for key, value in sorted(self.__dict__.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __str__(self):
        """返回配置对象的友好字符串表示"""
        return self.__repr__()


if __name__ == '__main__':
    config = DualChannelFusionConfig()
    print(config)
