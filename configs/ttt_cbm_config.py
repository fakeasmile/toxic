from pathlib import Path


class TTTCBMConfig:
    base_path = Path(__file__).parent.parent
    raw_data_path = base_path / "data" / "raw"
    processed_path = base_path / "data" / "processed"
    experiment_path = base_path / "experiments"
    models_path = base_path / "models"
    concept_path = raw_data_path / "adjective" / "discovered_concepts.csv"

    seed = 1
    use_deterministic = False

    plm_name = "chinese-roberta-wwm-ext"
    plm_hidden_size = 768
    max_length = 128
    freeze_plm = False
    plm_lr = 2e-5

    num_concepts = 56
    num_classes = 2
    dropout = 0.3
    concept_loss_weight = 0.1
    use_residual = True

    ttt_enabled = True
    ttt_lr = 1e-5
    ttt_steps = 1
    ttt_mlm_mask_ratio = 0.15

    batch_size = 16
    epochs = 30
    lr = 1e-4
    weight_decay = 0.01
    warmup_ratio = 0.1
    patience = 5

    llm_model_name = "Qwen2.5-7B-Instruct-AWQ"

    def __repr__(self):
        attrs = {k: v for k, v in vars(type(self)).items() if not k.startswith("_")}
        lines = [f"  {k}={v!r}" for k, v in attrs.items()]
        return f"{self.__class__.__name__}(\n" + ",\n".join(lines) + "\n)"

    def __str__(self):
        return self.__repr__()
