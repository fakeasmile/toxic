from pathlib import Path


class KIPCConfig:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.raw_data_path = self.base_path / "data" / "raw"
        self.processed_path = self.base_path / "data" / "processed"
        self.models_path = self.base_path / "models"
        self.experiment_path = self.base_path / "experiments"

        self.roberta_name = "chinese-roberta-wwm-ext"
        self.roberta_path = self.models_path / self.roberta_name
        self.sentence_model_name = "bge-base-zh-v1.5"
        self.sentence_model_path = self.models_path / self.sentence_model_name

        self.seed = 1
        self.use_deterministic = False

        self.dataset_name = "TOXICN"

        self.topic_map = {"race": 0, "gender": 1, "region": 2, "lgbt": 3, "none": 4}
        self.num_topics = 5
        self.num_expressions = 3
        self.num_targets = 5

        self.max_len = 128
        self.batch_size = 16
        self.epochs = 30
        self.lr_backbone = 2e-5
        self.lr_head = 1e-3
        self.patience = 10
        self.lambda_topic = 0.3
        self.lambda_expression = 0.3

        self.use_dual_encoder = True
        self.use_coded_terms = True
        self.use_homophone = True
        self.use_multitask = True
        self.use_platform = True

        self.num_coded_terms = 200
        self.coded_term_emb_dim = 64
        self.platform_emb_dim = 32

        self.homo_dict_path = self.raw_data_path / "adjective" / "homophone_map.json"
        self.coded_terms_path = self.raw_data_path / "adjective" / "coded_terms.json"

    def __repr__(self):
        lines = [f"{self.__class__.__name__}:"]
        for key, value in sorted(self.__dict__.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()
