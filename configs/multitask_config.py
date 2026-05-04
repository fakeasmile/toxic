from pathlib import Path


class MultiTaskConfig():
    def __init__(self):
        # ========== 路径相关配置 ==========
        self.base_path = Path(__file__).parent.parent
        self.raw_data_path = self.base_path / "data" / "raw"
        self.processed_path = self.base_path / "data" / "processed"
        self.bert_path = self.base_path / "models" / "bert-base-chinese"
        self.experiment_path = self.base_path / "experiments_multitask"

        # ========== 数据集配置 ==========
        self.dataset_name = "TOXICN"
        self.model_name = "Qwen2.5-7B-Instruct-AWQ"
        self.template = "likert"

        # ========== 随机种子配置 ==========
        self.seed = 1
        self.use_deterministic = False

        # ========== 训练超参数 ==========
        self.batch_size = 16
        self.epochs = 5
        self.learning_rate = 2e-5
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        self.max_seq_length = 128

        # ========== 模型参数 ==========
        self.dropout_rate = 0.1
        self.patience = 2
        self.concept_loss_weight = 0.1

    @property
    def train_path(self):
        return self.raw_data_path / self.dataset_name / "train.json"

    @property
    def test_path(self):
        return self.raw_data_path / self.dataset_name / "test.json"

    @property
    def train_concept_path(self):
        return self.processed_path / self.dataset_name / self.model_name / self.template / "concept_train.json"

    @property
    def test_concept_path(self):
        return self.processed_path / self.dataset_name / self.model_name / self.template / "concept_test.json"

    def __repr__(self):
        lines = [f"{self.__class__.__name__}:"]
        for key, value in sorted(self.__dict__.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()
