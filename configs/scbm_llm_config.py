from pathlib import Path


class SCBMLLMConfig:
    base_path = Path(__file__).parent.parent
    raw_data_path = base_path / "data" / "raw"
    processed_path = base_path / "data" / "processed"
    experiment_path = base_path / "experiments"
    models_path = base_path / "models"
    concept_path = raw_data_path / "adjective" / "discovered_concepts.csv"

    seed = 1
    use_deterministic = False

    llm_model_name = "Qwen2.5-7B-Instruct-AWQ"
    max_length = 128

    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    num_concepts = 56
    num_classes = 2
    concept_layer_idx = 20
    concept_loss_weight = 0.1
    use_residual = True
    use_4bit = True

    soft_label_weight = 0.5
    soft_label_temperature = 2.0

    batch_size = 4
    epochs = 10
    lr = 1e-4
    weight_decay = 0.01
    warmup_ratio = 0.1
    patience = 3
    gradient_accumulation_steps = 4
    max_grad_norm = 1.0

    def __repr__(self):
        attrs = {k: v for k, v in vars(type(self)).items() if not k.startswith("_")}
        lines = [f"  {k}={v!r}" for k, v in attrs.items()]
        return f"SCBMLLMConfig(\n" + ",\n".join(lines) + "\n)"

    def __str__(self):
        return self.__repr__()
