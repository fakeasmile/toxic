import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


class ConceptBottleneckLayer(nn.Module):
    def __init__(self, hidden_dim, num_concepts):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_concepts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        return self.sigmoid(self.fc(h))


class SCBMLLMModel(nn.Module):
    def __init__(
        self,
        model_name="Qwen2.5-7B-Instruct-AWQ",
        num_concepts=56,
        num_classes=2,
        concept_layer_idx=20,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        concept_loss_weight=0.1,
        soft_label_weight=0.5,
        soft_label_temperature=2.0,
        use_residual=True,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.concept_layer_idx = concept_layer_idx
        self.concept_loss_weight = concept_loss_weight
        self.soft_label_weight = soft_label_weight
        self.soft_label_temperature = soft_label_temperature
        self.use_residual = use_residual

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        hidden_dim = self.llm.config.hidden_size

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.llm = get_peft_model(self.llm, lora_config)

        self.concept_bottleneck = ConceptBottleneckLayer(hidden_dim, num_concepts)

        if use_residual:
            self.classifier = nn.Linear(hidden_dim + num_concepts, num_classes)
        else:
            self.classifier = nn.Linear(num_concepts, num_classes)

        self.ce_loss = nn.CrossEntropyLoss()
        self.concept_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        self._concept_hidden = None

        self._register_hook()

    def _register_hook(self):
        model = self.llm.base_model.model.model if hasattr(self.llm.base_model, 'model') else self.llm
        layers = model.model.layers

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self._concept_hidden = output[0][:, -1, :]
            else:
                self._concept_hidden = output[:, -1, :]

        if self.concept_layer_idx < len(layers):
            layers[self.concept_layer_idx].register_forward_hook(hook_fn)

    def forward(self, input_ids, attention_mask=None, labels=None, soft_labels=None, concept_labels=None):
        self._concept_hidden = None

        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits_all = outputs.logits

        last_token_logits = logits_all[:, -1, :]

        if self._concept_hidden is not None:
            concept_input = self._concept_hidden
        else:
            hidden_states = outputs.hidden_states
            if hidden_states is not None:
                layer_idx = min(self.concept_layer_idx + 1, len(hidden_states) - 1)
                concept_input = hidden_states[layer_idx][:, -1, :]
            else:
                concept_input = last_token_logits

        concept_probs = self.concept_bottleneck(concept_input)

        if self.use_residual:
            classifier_input = torch.cat([concept_input, concept_probs], dim=1)
        else:
            classifier_input = concept_probs

        logits = self.classifier(classifier_input)

        loss = None
        if labels is not None:
            ce = self.ce_loss(logits, labels)
            loss = ce

            if soft_labels is not None:
                T = self.soft_label_temperature
                log_probs = nn.functional.log_softmax(logits / T, dim=1)
                soft_targets = nn.functional.softmax(soft_labels / T, dim=1)
                kl = self.kl_loss(log_probs, soft_targets) * (T * T)
                loss = loss + self.soft_label_weight * kl

            if concept_labels is not None:
                mse = self.concept_loss(concept_probs, concept_labels)
                loss = loss + self.concept_loss_weight * mse

        if loss is not None:
            return logits, concept_probs, loss
        return logits, concept_probs
