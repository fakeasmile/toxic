import torch
import torch.nn as nn
from transformers import AutoModel


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = lambda_adv
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None


def grad_reverse(x, lambda_adv=1.0):
    return GradientReversalLayer.apply(x, lambda_adv)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class TADCLClassifier(nn.Module):
    def __init__(self, roberta_path, num_coded_terms=72, num_platforms=2,
                 coded_emb_dim=64, platform_emb_dim=32, use_dual_encoder=True,
                 use_coded_terms=True, use_homophone=True, use_multitask=True,
                 use_platform=True, num_topics=4, num_expressions=4,
                 use_supcon=True, projection_dim=128,
                 use_adversary=True):
        super().__init__()
        self.use_dual_encoder = use_dual_encoder
        self.use_coded_terms = use_coded_terms
        self.use_homophone = use_homophone
        self.use_multitask = use_multitask
        self.use_platform = use_platform
        self.use_supcon = use_supcon
        self.use_adversary = use_adversary

        self.roberta = AutoModel.from_pretrained(str(roberta_path))
        hidden_size = self.roberta.config.hidden_size

        if use_dual_encoder:
            fused_dim = hidden_size * 2
        else:
            fused_dim = hidden_size

        if use_coded_terms:
            self.coded_proj = nn.Sequential(
                nn.Linear(num_coded_terms, coded_emb_dim),
                nn.ReLU(),
                nn.Linear(coded_emb_dim, fused_dim),
            )

        if use_platform:
            self.platform_emb = nn.Embedding(num_platforms, platform_emb_dim)
            self.platform_gate_proj = nn.Linear(platform_emb_dim, fused_dim)

        self.dropout = nn.Dropout(0.1)
        self.toxic_head = nn.Linear(fused_dim, 2)

        if use_multitask:
            self.topic_head = nn.Linear(fused_dim, num_topics)
            self.expression_head = nn.Linear(fused_dim, num_expressions)

        if use_supcon:
            self.projection_head = ProjectionHead(fused_dim, fused_dim // 2, projection_dim)

        if use_adversary:
            self.topic_adversary = nn.Sequential(
                nn.Linear(fused_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_topics),
            )

    def encode(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0]
        return cls_hidden

    def forward(self, input_ids_orig, attention_mask_orig,
                input_ids_rest=None, attention_mask_rest=None,
                coded_multi_hot=None, platform_ids=None,
                lambda_adv=1.0):
        h_orig = self.encode(input_ids_orig, attention_mask_orig)

        if self.use_dual_encoder and input_ids_rest is not None:
            h_rest = self.encode(input_ids_rest, attention_mask_rest)
            h_fused = torch.cat([h_orig, h_rest], dim=-1)
        else:
            h_fused = h_orig

        if self.use_coded_terms and coded_multi_hot is not None:
            coded_signal = self.coded_proj(coded_multi_hot)
            h_fused = h_fused + coded_signal

        if self.use_platform and platform_ids is not None:
            platform_gate = 1 + torch.sigmoid(self.platform_gate_proj(self.platform_emb(platform_ids)))
            h_fused = h_fused * platform_gate

        h_fused = self.dropout(h_fused)

        toxic_logits = self.toxic_head(h_fused)

        result = {"toxic_logits": toxic_logits, "h_fused": h_fused}

        if self.use_multitask:
            result["topic_logits"] = self.topic_head(h_fused)
            result["expression_logits"] = self.expression_head(h_fused)

        if self.use_supcon:
            result["projected"] = self.projection_head(h_fused)

        if self.use_adversary:
            reversed_h = grad_reverse(h_fused, lambda_adv)
            result["adv_topic_logits"] = self.topic_adversary(reversed_h)

        return result
