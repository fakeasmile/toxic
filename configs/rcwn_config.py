from pathlib import Path


class RCWNConfig:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.raw_data_path = self.base_path / "data" / "raw"
        self.processed_path = self.base_path / "data" / "processed"
        self.models_path = self.base_path / "models"
        self.experiment_path = self.base_path / "experiments"

        self.seed = 1
        self.use_deterministic = False

        self.dataset_name = "TOXICN"
        self.plm_name = "chinese-roberta-wwm-ext"

        self.num_concepts = 40
        self.concept_vocab_path = self.raw_data_path / "adjective" / "discovered_concepts.csv"

        self.plm_frozen = True
        self.plm_lr = 2e-5
        self.cw_lr = 1e-3
        self.head_lr = 1e-3

        self.batch_size = 32
        self.epochs = 30
        self.max_lr = 2e-5
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01

        self.dropout = 0.3
        self.residual_hidden_dim = 256
        self.patience = 5

        self.lambda_align = 1.0
        self.lambda_ortho = 0.1
        self.alpha = 0.5

        self.max_seq_length = 128

        self.llm_model_name = "Qwen2.5-7B-Instruct-AWQ"
        self.llm_quantization = "awq"
        self.llm_gpu_memory_utilization = 0.85
        self.llm_temperature = 2.0

        self.discover_sample_size = 200
        self.discover_concepts_per_round = 15
        self.discover_rounds = 3

    def __repr__(self):
        lines = [f"{self.__class__.__name__}:"]
        for key, value in sorted(self.__dict__.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


if __name__ == '__main__':
    config = RCWNConfig()
    print(config)
