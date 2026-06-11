"""CB-LLM-CN配置类

Concept Bottleneck LLM for Chinese Toxic Language Detection
基于CB-LLM (ICLR 2025)方法，针对中文有害言论检测场景适配。
"""

from pathlib import Path


class CBLLMConfig:
    """CB-LLM-CN训练配置

    继承MLPConfig的基本路径和种子配置，新增CB-LLM-CN特有的参数。
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
        self.backbone = "bge"  # backbone模式: bge/qwen/concept_only

        # ========== ACS概念评分配置 ==========
        self.embedding_model_name = "bge-base-zh-v1.5"  # BGE中文嵌入模型
        self.acs_batch_size = 64  # 嵌入模型推理批次大小

        # ========== CBL概念瓶颈层配置 ==========
        self.cbl_hidden_dim = 256  # CBL中间层维度
        self.cbl_dropout = 0.3  # CBL Dropout比率

        # ========== 稀疏预测层配置 ==========
        self.sparse_l1_weight = 0.01  # L1稀疏正则化权重
        self.prediction_dropout = 0.3  # 预测层Dropout
        self.topk = 0  # TopK稀疏激活数量，0=不使用
        self.use_residual = False  # 是否使用残差连接

        # ========== 训练超参数 ==========
        self.batch_size = 32
        self.epochs = 200

        # OneCycleLR 学习率调度器参数
        self.max_lr = 1e-3
        self.pct_start = 0.2
        self.div_factor = 25.0
        self.final_div_factor = 10000.0
        self.anneal_strategy = 'cos'

        self.patience = 30  # 早停耐心值

        # ========== MLP参数（concept_only模式）==========
        self.dropout_rate = 0.5  # MLP Dropout比率
        self.hidden_features = 96  # MLP隐藏层维度

    def __repr__(self):
        """返回配置对象的字符串表示"""
        lines = [f"{self.__class__.__name__}:"]
        for key, value in sorted(self.__dict__.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


if __name__ == '__main__':
    config = CBLLMConfig()
    print(config)
