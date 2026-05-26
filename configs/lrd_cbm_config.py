from pathlib import Path


class LRDCBMConfig():
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.raw_data_path = self.base_path / "data" / "raw"
        self.processed_path = self.base_path / "data" / "processed"
        self.experiment_path = self.base_path / "experiments"
        self.concept_path = self.raw_data_path / "adjective" / "discovered_concepts.csv"
        self.rationale_path = self.processed_path

        self.seed = 1
        self.use_deterministic = False

        self.plm_name = "chinese-roberta-wwm-ext"
        self.plm_hidden_size = 768
        self.max_length = 128
        self.rationale_max_length = 256
        self.freeze_plm = False
        self.plm_lr = 2e-5
        self.share_plm = True

        self.num_concepts = 56
        self.num_classes = 2
        self.dropout = 0.3
        self.concept_loss_weight = 0.1
        self.use_rationale = True

        self.batch_size = 16
        self.epochs = 30
        self.lr = 1e-4
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.patience = 5

        self.llm_model_name = "Qwen2.5-7B-Instruct-AWQ"
        self.llm_batch_size = 32
        self.llm_max_tokens = 512

    def __repr__(self):
        lines = [f"{self.__class__.__name__}:"]
        for key, value in sorted(self.__dict__.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()


if __name__ == '__main__':
    config = LRDCBMConfig()
    print(config)
