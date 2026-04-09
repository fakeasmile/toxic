from pathlib import Path

class BaseConfig():
    def __init__(self):
        self.base_path = Path(__file__).parent.parent  # 基目录

        self.train_path = self.base_path / "data" / "raw" / "rawtrain.json"  # 训练数据集路径
        self.test_path = self.base_path / "data" / "raw" / "rawtest.json"  # 测试集路径
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



        self.train_ratio = 0.9  # 超参数搜索中训练集划分的比例
        self.train_pct_start = 0.15  # 在超参数搜索中，学习率上升期（不触发早停）的比例
        self.train_patience = 2  # 在超参数搜索中，早停耐心值

        self.use_deterministic = False  # 启用确定性算法以确保可复现




if __name__ == '__main__':
    config = BaseConfig()
    print(config.base_path)