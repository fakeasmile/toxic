import copy
import torch
import torch.nn as nn
from transformers import AutoModel


class ConceptBottleneckLayer(nn.Module):
    def __init__(self, hidden_dim, num_concepts):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_concepts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        h_normed = self.layer_norm(h)
        return self.sigmoid(self.fc(h_normed))


class TTTCBMModel(nn.Module):
    def __init__(
        self,
        plm_name="chinese-roberta-wwm-ext",
        num_concepts=56,
        num_classes=2,
        dropout=0.3,
        concept_loss_weight=0.1,
        use_residual=True,
        ttt_enabled=True,
        ttt_lr=1e-5,
        ttt_steps=1,
        ttt_mlm_mask_ratio=0.15,
    ):
        super().__init__()
        self.plm = AutoModel.from_pretrained(plm_name)
        plm_hidden = self.plm.config.hidden_size

        self.num_concepts = num_concepts
        self.concept_loss_weight = concept_loss_weight
        self.use_residual = use_residual
        self.ttt_enabled = ttt_enabled
        self.ttt_lr = ttt_lr
        self.ttt_steps = ttt_steps
        self.ttt_mlm_mask_ratio = ttt_mlm_mask_ratio

        self.dropout = nn.Dropout(dropout)
        self.concept_bottleneck = ConceptBottleneckLayer(plm_hidden, num_concepts)

        if use_residual:
            self.classifier = nn.Linear(plm_hidden + num_concepts, num_classes)
        else:
            self.classifier = nn.Linear(num_concepts, num_classes)

        self.ce_loss = nn.CrossEntropyLoss()
        self.concept_loss_fn = nn.MSELoss()

        vocab_size = self.plm.config.vocab_size
        self.mlm_head = nn.Sequential(
            nn.Linear(plm_hidden, plm_hidden),
            nn.GELU(),
            nn.LayerNorm(plm_hidden),
            nn.Linear(plm_hidden, vocab_size),
        )

        self._ttt_original_state = None

    def forward(self, input_ids, attention_mask, labels=None, concept_labels=None):
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        h = outputs.last_hidden_state[:, 0, :]
        h_drop = self.dropout(h)

        concept_probs = self.concept_bottleneck(h_drop)

        if self.use_residual:
            classifier_input = torch.cat([h_drop, concept_probs], dim=1)
        else:
            classifier_input = concept_probs

        logits = self.classifier(classifier_input)

        loss = None
        if labels is not None:
            ce = self.ce_loss(logits, labels)
            loss = ce
            if concept_labels is not None:
                mse = self.concept_loss_fn(concept_probs, concept_labels)
                loss = loss + self.concept_loss_weight * mse

        if loss is not None:
            return logits, concept_probs, loss
        return logits, concept_probs

    def ttt_adapt(self, input_ids, attention_mask):
        if not self.ttt_enabled:
            return

        ttt_params = list(self.concept_bottleneck.parameters()) + list(self.classifier.parameters())

        self._ttt_original_state = copy.deepcopy({
            k: v.clone() for k, v in self.state_dict().items()
        })

        labels_mlm = input_ids.clone()
        probability_matrix = torch.full(labels_mlm.shape, self.ttt_mlm_mask_ratio)
        special_tokens_mask = attention_mask == 0
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        mlm_mask = torch.bernoulli(probability_matrix).bool()
        labels_mlm[~mlm_mask] = -100

        masked_input_ids = input_ids.clone()
        mask_token_id = self.plm.config.vocab_size - 1
        masked_input_ids[mlm_mask] = mask_token_id

        ttt_optimizer = torch.optim.SGD(ttt_params, lr=self.ttt_lr)

        for _ in range(self.ttt_steps):
            outputs = self.plm(input_ids=masked_input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            mlm_logits = self.mlm_head(hidden_states)

            mlm_loss = nn.functional.cross_entropy(
                mlm_logits.view(-1, mlm_logits.size(-1)),
                labels_mlm.view(-1),
                ignore_index=-100,
            )

            if mlm_loss > 0:
                ttt_optimizer.zero_grad()
                mlm_loss.backward()
                ttt_optimizer.step()

    def ttt_restore(self):
        if self._ttt_original_state is not None:
            self.load_state_dict(self._ttt_original_state)
            self._ttt_original_state = None
