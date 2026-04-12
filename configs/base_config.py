from pathlib import Path

class BaseConfig():
    def __init__(self):
        self.base_path = Path(__file__).parent.parent  # 基目录

        self.dataset_name = "TOXICN"  # 数据集名称（TOXICN/COLD）
        self.train_path = self.base_path / "data" / "raw" / self.dataset_name / "train.json"  # 训练数据集路径
        self.test_path = self.base_path / "data" / "raw" / self.dataset_name / "test.json"  # 测试集路径
        self.bert_path = self.base_path / "models" / "bert-base-chinese"  # Bert模型路径
        self.attack_stance_dict_path = self.base_path / "data" / "raw" / "attack_stance.json"  # 攻击立场词典路径
        self.dirty_lexicon_path = self.base_path / "data" / "raw" / "lexicon"  # 脏词词典路径
        self.experiment_path = self.base_path / "experiments"  # 实验结果保存目录
        self.processed_path = self.base_path / "data" / "processed"  # 数据处理保存路径
        self.adjective_path = self.base_path / "data" / "raw" / "adjective" / "toxic_adjectives.csv"  # 形容词词典路径
        self.models_path = self.base_path / "models"  # LLM模型路径

        self.seed = 1  # 随机种子
        self.epochs = 8
        self.dropout_rate = 0.5  # FNN中dropout率
        self.freeze_bert_layers = 0  # 冻结BERT模型前freeze_bert_layers层
        self.num_toxic_types=6  # 词典lexicon中毒性类别数

        # 嵌入层投影类型: "linear"/"linear_norm"/"linear_act_norm"
        self.proj_type = "linear"

        # 训练超参数
        self.batch_size = 64  # 批次大小
        self.lr = 1e-5  # 学习率
        self.max_len = 80  # 最大序列长度
        self.weight_decay = 0  # 权重衰减（当前未使用）

        self.train_ratio = 0.9  # 超参数搜索中训练集划分的比例
        self.train_pct_start = 0.15  # 在超参数搜索中，学习率上升期（不触发早停）的比例
        self.train_patience = 2  # 在超参数搜索中，早停耐心值

        self.use_deterministic = True  # 启用确定性算法以确保可复现

        # 可视化配置
        self.fig_size = (12, 7)  # 图表大小




if __name__ == '__main__':
    config = BaseConfig()
    print(config.base_path)