from pathlib import Path


class MLPConfig():
    def __init__(self):
        # ========== 路径相关配置（公共配置）==========
        self.base_path = Path(__file__).parent.parent  # 基目录
        
        self.train_path = self.base_path / "data" / "raw" / "rawtrain.json"  # 训练数据集路径
        self.test_path = self.base_path / "data" / "raw" / "_test.json"  # 测试集路径
        self.experiment_path = self.base_path / "experiments"  # 实验结果保存目录
        self.processed_path = self.base_path / "data" / "processed"  # 数据处理保存路径

        # ========== 随机种子配置（公共配置）==========
        self.seed = 1  # 随机种子
        self.use_deterministic = False  # 启用确定性算法以确保可复现

        # ========== MLP 训练超参数 ==========
        self.batch_size = 16  # 批次大小
        self.epochs = 100  # 训练轮数
        
        # OneCycleLR 学习率调度器参数
        self.max_lr = 1e-3  # 峰值学习率
        self.pct_start = 0.2  # Warmup 占总步数的比例
        self.div_factor = 25.0  # 初始学习率 = max_lr / div_factor
        self.final_div_factor = 10000.0  # 最终学习率 = max_lr / final_div_factor
        self.anneal_strategy = 'cos'  # Warmup 后余弦衰减

        # ========== MLP 模型参数 ==========
        self.dropout_rate = 0.2  # Dropout 比率
        self.hidden_features = 96  # 隐藏层特征维度


if __name__ == '__main__':
    config = MLPConfig()
    print(config.base_path)
