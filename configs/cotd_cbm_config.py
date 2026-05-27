from pathlib import Path


class CoTDCBMConfig:
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
    soft_label_weight = 0.5
    soft_label_temperature = 2.0
    use_residual = True

    batch_size = 16
    epochs = 30
    lr = 1e-4
    weight_decay = 0.01
    warmup_ratio = 0.1
    patience = 5

    llm_model_name = "Qwen2.5-7B-Instruct-AWQ"

    def __repr__(self):
        attrs = {k: v for k, v in vars(self.__class__).items() if not k.startswith("_")}
        lines = [f"  {k} = {v!r}" for k, v in attrs.items()]
        return f"CoTDCBMConfig(\n" + ",\n".join(lines) + "\n)"

    def __str__(self):
        return self.__repr__()
