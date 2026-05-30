from pathlib import Path


class KISCBConfig():
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.raw_data_path = self.base_path / "data" / "raw"
        self.processed_path = self.base_path / "data" / "processed"
        self.models_path = self.base_path / "models"
        self.experiment_path = self.base_path / "experiments"

        self.homo_graph_path = self.raw_data_path / "homo_graph.json"
        self.toxic_lexicon_path = self.raw_data_path / "toxic_lexicon.json"
        self.coded_terms_path = self.raw_data_path / "adjective" / "discovered_concepts.csv"

        self.seed = 1
        self.use_deterministic = False

        self.plm_name = "chinese-roberta-wwm-ext"
        self.max_length = 128

        self.num_platforms = 2
        self.num_topics = 4

        self.num_targets = 5
        self.num_strategies = 4
        self.num_intents = 5
        self.num_tones = 4

        self.concept_emb_dim = 128
        self.dropout_rate = 0.3

        self.batch_size = 16
        self.epochs = 30
        self.plm_lr = 2e-5
        self.max_lr = 1e-4
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.patience = 5

        self.lambda_toxic = 1.0
        self.lambda_target = 0.5
        self.lambda_strategy = 0.5
        self.lambda_intent = 0.3
        self.lambda_tone = 0.3
        self.lambda_consistency = 0.1

        self.stage1_epochs = 5
        self.stage3_epochs = 10

    def __repr__(self):
        lines = [f"{self.__class__.__name__}:"]
        for key, value in sorted(self.__dict__.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


if __name__ == '__main__':
    config = KISCBConfig()
    print(config)
