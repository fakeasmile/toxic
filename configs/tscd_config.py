from pathlib import Path


class TSCDConfig:
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

        self.num_concepts = 56
        self.concept_vocab_path = self.raw_data_path / "adjective" / "discovered_concepts.csv"

        # Stage 1: PLM + SCL + FGM
        self.s1_batch_size = 32
        self.s1_epochs = 10
        self.s1_lr = 2e-5
        self.s1_warmup_ratio = 0.1
        self.s1_weight_decay = 0.01
        self.s1_patience = 3
        self.s1_dropout = 0.3
        self.s1_hidden_dim = 256
        self.s1_max_seq_length = 128

        self.use_scl = True
        self.scl_temperature = 0.07
        self.lambda_scl = 0.1

        self.use_fgm = True
        self.fgm_epsilon = 0.5
        self.lambda_adv = 0.5

        # Stage 2: Concept Distillation
        self.s2_batch_size = 32
        self.s2_epochs = 30
        self.s2_lr = 1e-3
        self.s2_weight_decay = 0.01
        self.s2_patience = 10
        self.s2_concept_hidden_dim = 128
        self.s2_max_seq_length = 128
        self.lambda_concept = 1.0

        self.llm_model_name = "Qwen2.5-7B-Instruct-AWQ"

    def __repr__(self):
        lines = [f"{self.__class__.__name__}:"]
        for key, value in sorted(self.__dict__.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def __str__(self):
        return self.__repr__()
